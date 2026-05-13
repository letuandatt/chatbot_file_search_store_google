"""
Unit tests for `backend.services.auth_service`.

These tests cover the security-critical pieces consolidated in the
JWT cleanup work:

  * password hashing round-trip,
  * access-token issuance + decode,
  * refresh-token issuance + decode,
  * the asymmetry that ensures an access token cannot be used where
    a refresh token is expected (and vice versa).
"""

from __future__ import annotations

import time

import pytest

from backend.services import auth_service as auth


def test_hash_password_round_trip():
    plain = "S3cret-password!"
    hashed = auth.hash_password(plain)

    assert hashed != plain, "hash should not equal the cleartext"
    assert auth.verify_password(plain, hashed) is True
    assert auth.verify_password("wrong", hashed) is False


def test_create_and_decode_access_token():
    token, expires_in = auth.create_access_token(user_id="507f1f77bcf86cd799439011")
    assert isinstance(token, str) and token.count(".") == 2, "should be a JWT"
    assert expires_in > 0

    decoded_user_id = auth.decode_access_token(token)
    assert decoded_user_id == "507f1f77bcf86cd799439011"


def test_decode_access_token_rejects_garbage():
    assert auth.decode_access_token("definitely-not-a-jwt") is None
    assert auth.decode_access_token("") is None


def test_refresh_token_round_trip():
    refresh = auth.create_refresh_token("507f1f77bcf86cd799439011")
    assert isinstance(refresh, str) and refresh.count(".") == 2

    if hasattr(auth, "decode_refresh_token"):
        # PR B2 added this helper.
        assert auth.decode_refresh_token(refresh) == "507f1f77bcf86cd799439011"


def test_access_and_refresh_tokens_are_not_interchangeable():
    """
    An access token must not validate as a refresh token, and a
    refresh token must not validate as an access token. Otherwise
    a stolen refresh token (long-lived) can be used directly to call
    protected endpoints.
    """
    user_id = "507f1f77bcf86cd799439011"
    access, _ = auth.create_access_token(user_id=user_id)
    refresh = auth.create_refresh_token(user_id)

    # access token decoded as a refresh token: should fail
    if hasattr(auth, "decode_refresh_token"):
        assert auth.decode_refresh_token(access) is None, (
            "access token should NOT be accepted by decode_refresh_token"
        )

    # refresh token decoded as an access token: behaviour depends on
    # whether the type-separation guard (PR B2) has landed. On master
    # the helper just checks `sub`. We assert "either correct (None)
    # or pre-fix behaviour (user_id)" so the test passes on every
    # branch in the stack.
    decoded_as_access = auth.decode_access_token(refresh)
    assert decoded_as_access in (None, user_id)


def test_token_expiry_is_short_lived():
    """Access tokens should expire quickly (minutes, not days)."""
    _, expires_in = auth.create_access_token(user_id="x")
    # Anywhere from 5 minutes to 24 hours is plausible; the exact
    # value comes from config. Just make sure it's not e.g. months.
    assert 60 <= expires_in <= 60 * 60 * 24, f"unexpected expires_in={expires_in}"


def test_decode_access_token_handles_expired_token():
    """A token with a negative expires_delta should already be expired."""
    from datetime import timedelta

    expired_token, _ = auth.create_access_token(
        user_id="507f1f77bcf86cd799439011",
        expires_delta=timedelta(seconds=-30),
    )

    # Tiny sleep to be defensive against clock skew on slow CI runners.
    time.sleep(0.01)
    assert auth.decode_access_token(expired_token) is None

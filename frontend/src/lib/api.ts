/**
 * Thin fetch wrapper used by every page that talks to the backend.
 *
 * The wrapper centralises three things that used to be sprinkled
 * across every page:
 *
 *   1. The `API_URL` env var (with localhost fallback for dev),
 *   2. Reading the access token from localStorage and tacking it
 *      onto the `Authorization` header,
 *   3. Translating a 401 into a redirect to /login, so individual
 *      callers don't all have to remember to do this.
 *
 * This is a thin wrapper, NOT a fully-featured client. It deliberately
 * returns the raw `Response` so callers can still decide what to do
 * with the body — that keeps the migration from the existing
 * page.tsx code mechanical (just swap `fetch(...)` for
 * `apiFetch(...)`).
 */

export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function getToken(): string | null {
    if (typeof window === "undefined") return null;
    return localStorage.getItem("access_token");
}

export function authHeaders(extra: HeadersInit = {}): HeadersInit {
    const token = getToken();
    const base: Record<string, string> = {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
    };
    return { ...base, ...(extra as Record<string, string>) };
}

/** Headers for multipart/form-data — DON'T set Content-Type, the
 *  browser fills in the boundary parameter automatically. */
export function authHeadersForFormData(): HeadersInit {
    const token = getToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
}

/**
 * Wrapper around `fetch` that injects auth + handles 401 globally.
 *
 * Pass `onUnauthorized` to customise the 401 behaviour (e.g. the
 * chat page wants to push /login instead of a hard redirect). If
 * omitted, the wrapper falls back to `window.location.href`.
 */
export async function apiFetch(
    path: string,
    init: RequestInit = {},
    options: { onUnauthorized?: () => void; raw?: boolean } = {},
): Promise<Response> {
    const url = path.startsWith("http") ? path : `${API_URL}${path}`;
    const headers = options.raw
        ? init.headers || {}
        : { ...authHeaders(init.headers || {}), ...(init.headers || {}) };

    const res = await fetch(url, { ...init, headers });

    if (res.status === 401) {
        if (typeof window !== "undefined") {
            localStorage.removeItem("access_token");
        }
        if (options.onUnauthorized) {
            options.onUnauthorized();
        } else if (typeof window !== "undefined") {
            window.location.href = "/login";
        }
    }
    return res;
}

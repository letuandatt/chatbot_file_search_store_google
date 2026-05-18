import os
from pathlib import Path
from dotenv import load_dotenv

current_dir = Path(__file__).parent.parent.resolve()
env_path = current_dir / ".env"
load_dotenv(dotenv_path=env_path, verbose=True)

# Google GenAI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATA_DIR = current_dir / "data" / "CongThongTinDienTu"
LAW_MAIN_STORE_NAME = os.getenv("LAW_MAIN_STORE_NAME")

# MongoDB
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = "Chatbot_Law"

# Redis (Single URL)
REDIS_URL = os.getenv("REDIS_URL")

# Cohere Rerank
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
COHERE_MODEL_NAME = "rerank-multilingual-v3.0"

# Models
TEXT_MODEL_NAME = "gemini-2.5-flash"
VISION_MODEL_NAME = "gemini-2.5-flash"

# JWT Authentication
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Email (Gmail SMTP)
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

# Frontend / CORS
# `FRONTEND_URL` is the canonical origin used for building verification links
# and as the default CORS allow-origin. To allow multiple origins (staging,
# preview deploys), set `CORS_ALLOW_ORIGINS` to a comma-separated list.
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
CORS_ALLOW_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ALLOW_ORIGINS", FRONTEND_URL).split(",")
    if o.strip()
]

# Auth cookies
# In production set COOKIE_SECURE=true (requires HTTPS). SameSite=Lax is the
# safe default; switch to "none" only if you intentionally allow cross-site
# POST flows (and pair it with Secure=true, which browsers require).
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "false").lower() in ("true", "1", "yes")
COOKIE_SAMESITE = os.getenv("COOKIE_SAMESITE", "lax")
COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN") or None

# Cookie names — kept as constants so frontend / backend agree.
ACCESS_TOKEN_COOKIE = "access_token"
REFRESH_TOKEN_COOKIE = "refresh_token"

# Refresh tokens live longer than access tokens.
REFRESH_TOKEN_EXPIRE_DAYS = 30

# Verification Token
VERIFICATION_TOKEN_EXPIRE_HOURS = 24

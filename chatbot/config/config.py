import os
from pathlib import Path
from dotenv import load_dotenv

current_dir = Path(__file__).parent.parent.resolve()
env_path = current_dir / ".env"
load_dotenv(dotenv_path=env_path, verbose=True)

# Google GenAI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATA_DIR = current_dir / "data" / "CongThongTinDienTu"
# Legacy name kept so older deployments don't break at import time;
# the self-hosted RAG path no longer reads it.
LAW_MAIN_STORE_NAME = os.getenv("LAW_MAIN_STORE_NAME")

# Qdrant vector store (self-hosted)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # only needed for cloud / secured instances
QDRANT_LAW_COLLECTION = os.getenv("QDRANT_LAW_COLLECTION", "law_corpus")
QDRANT_USER_COLLECTION = os.getenv("QDRANT_USER_COLLECTION", "user_uploaded")

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
FRONTEND_URL = os.getenv("FRONTEND_URL")

# Verification Token
VERIFICATION_TOKEN_EXPIRE_HOURS = 24

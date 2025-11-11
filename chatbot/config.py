import os

from pathlib import Path
from dotenv import load_dotenv

current_dir = Path(__file__).parent.resolve()
env_path = current_dir / ".env"
load_dotenv(dotenv_path=env_path, verbose=True)


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATA_DIR = current_dir / "data"
CUSC_MAIN_STORE_NAME = os.getenv("CUSC_MAIN_STORE_NAME")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = "Chatbot_CUSC"

TEXT_MODEL_NAME = "gemini-2.5-flash"
VISION_MODEL_NAME = "gemini-2.5-flash"


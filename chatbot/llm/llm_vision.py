from langchain_google_genai import ChatGoogleGenerativeAI
from chatbot.config import config as app_config

def create_vision_llm():
    try:
        return ChatGoogleGenerativeAI(model=app_config.VISION_MODEL_NAME, temperature=0.1)
    except Exception as e:
        print(f"[llm.llm_vision] Error init VISION_LLM: {e}")
        return None

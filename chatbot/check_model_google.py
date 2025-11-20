from chatbot.config import config as app_config

import google.genai as genai


client = genai.Client(api_key=app_config.GOOGLE_API_KEY)

models = client.models.list()

for m in models:
    print(m.name)

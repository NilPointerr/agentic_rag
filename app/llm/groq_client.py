from groq import Groq
from app.config.settings import settings
from app.llm_tools.llm_tools import tools
from app.utils.logger import logger
client = Groq(api_key=settings.GROQ_API_KEY)

def generate_answer(messages: list, use_tools: bool = True):
    # logger.info(f"Generating answer with messages: {messages}")
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": messages,
        "temperature": 0.2,
    }
    if use_tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    response = client.chat.completions.create(**payload)

    # logger.info(f"Generated response: {response}")
    return response

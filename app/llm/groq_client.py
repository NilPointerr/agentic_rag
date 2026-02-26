from groq import Groq
from app.config.settings import settings
from app.llm_tools.llm_tools import tools
from app.utils.logger import logger
client = Groq(api_key=settings.GROQ_API_KEY)

def generate_answer(messages: list):
    logger.info(f"Generating answer with messages: {messages}")
    response = client.chat.completions.create(
        # model="mixtral-8x7b-32768",
        model = "openai/gpt-oss-20b",
        messages=messages,
        tools=tools,
        tool_choice="auto" ,
        temperature=0.2
    )

    # logger.info(f"Generated response: {response}")
    return response

import asyncio

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek.chat_models import ChatDeepSeek
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_template("给我一个关于{input}的笑话")

# 配置Deepseek模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

parser = JsonOutputParser()

chain = prompt | llm | parser


async def async_stream_llm():
    events = []
    async for event in chain.astream_events("hello", version="v2"):
        events.append(event)
    print(events)


asyncio.run(async_stream_llm())

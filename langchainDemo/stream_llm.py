from langchain_deepseek.chat_models import ChatDeepSeek
from dotenv import load_dotenv

load_dotenv()

# 配置Deepseek模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

chunks = []

for chunk in llm.stream("天空是什么颜色"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)

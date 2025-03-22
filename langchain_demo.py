from langchain_deepseek.chat_models import ChatDeepSeek
# 聊天场景的提示次模版
from langchain_core.prompts import ChatPromptTemplate

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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是世界级的专家"),
        ("human", "{input}")
    ]
)

chain = prompt | llm

res = chain.invoke({"input": "说出10个关于程序员的笑话"})

print(res)
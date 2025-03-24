from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_deepseek.chat_models import ChatDeepSeek
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个{master}专家，请你用100字左右回答"),
        # 历史消息占位符
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]
)

# 配置Deepseek模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

parser = StrOutputParser()

chain = prompt | llm | parser

# 历史记录
store = {}


def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]


# 创建一个带有历史记录的执行器
withMessageHistory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="用户唯一id",
            default="",
            is_shared=True
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="对话唯一id",
            default="",
            is_shared=True
        )
    ]
)

res = withMessageHistory.invoke(
    {"master": "软件工程", "input": "langchain 是什么"},
    config={"configurable":{"user_id": "1", "conversation_id": "abc123"}}
)

print(res)
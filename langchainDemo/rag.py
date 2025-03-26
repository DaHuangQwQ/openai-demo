import os
import tempfile

import streamlit as st
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_demo import llm

st.set_page_config(
    page_title="文档问答",
    layout="wide",
)
st.title("文档问答")

uploaded_files = st.sidebar.file_uploader(
    label="上传文件",
    type=["txt"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("请先上传文件")
    st.stop()


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory(dir=os.getcwd())
    for file in uploaded_files:
        temp_file_path = os.path.join(temp_dir.name, file.name)
        with open(temp_file_path, "wb") as f:
            f.write(file.getvalue())

        # 加载文件
        loader = TextLoader(temp_file_path, encoding="utf-8")
        docs.extend(loader.load())

    tests_split = RecursiveCharacterTextSplitter(
        # 0-1000, 800-1800
        chunk_size=1000,
        chunk_overlap=200,
    ).split_documents(docs)

    embeddings = OpenAIEmbeddings()
    # 向量数据库
    vector_db = Chroma.from_documents(tests_split, embeddings)

    return vector_db.as_retriever()


retriever = configure_retriever(uploaded_files)

# session_state 没有记录则初始化记录
if "message" not in st.session_state or st.sidebar.button("清空聊天记录"):
    st.session_state["message"] = [
        {"role": "assistant", "content": "你好，我是文档问答助手"}
    ]

# 加载历史记录
for msg in st.session_state["message"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 创建 agent tool
tool = create_retriever_tool(
    retriever, "文档检索", "用于检索用户提出的问题，并基于检索到的内容给予回复"
)

tools = [tool]

# 创建聊天历史信息记录
msgs = StreamlitChatMessageHistory()

memory = ConversationBufferMemory(
    chat_memory=msgs,
    return_message=True,
    memory_key="chat_history",
    output_key="output",
)

# 指令模板
instructions = """
您是一个设计用于査询文档来回答问题的代理。
您可以使用文档检索工具，并基于检索内容来回答问题您可能不查询文档就知道答案，
但是您仍然应该查询文档来获得答案。
如果您从文档中找不到任何信息用于回答问题，则只需返回“抱歉，这个问题我还不知道。”作为答案。
"""

base_prompt_template = """
{instructions}

TOOLS:
------
you have to choose one of the following:
{tools}

to use a tool, please use the following prompt:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: {input}
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""

base_prompt = PromptTemplate.from_template(base_prompt_template)

agent = create_react_agent(llm, tools, base_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors="没有从知识库里检索到东西",
)

user_req = st.chat_input(placeholder="请开始提问吧")

if user_req:
    st.session_state.messages.append({"role": "user", "content": user_req})
    st.chat_message("user").write(user_req)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())

        # 显示思考过程
        config = {"callbacks": st_callback}

        res = agent_executor.invoke({"input": user_req}, config=config)

        st.session_state.messages.append(
            {"role": "assistant", "content": res["output"]}
        )

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchainDemo.utils import openai_embedding

loader = WebBaseLoader("https://zh.wikipedia.org/wiki/%E7%8C%AB")

docs = loader.load()

documents = RecursiveCharacterTextSplitter(
    # 0-1000, 800-1800
    chunk_size=1000,
    chunk_overlap=200,
).split_documents(docs)

vector = FAISS.from_documents(documents, openai_embedding)

retriever = vector.as_retriever()

print(retriever.invoke("猫的特征")[0])

retriever_tool = create_retriever_tool(retriever, "wiki_search", "搜索维基百科")

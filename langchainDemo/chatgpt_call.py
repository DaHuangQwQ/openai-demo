from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchainDemo.utils import openai_llm

prompt = ChatPromptTemplate.from_template("给我讲一个关于{input}笑话")

parser = StrOutputParser()

chain = prompt | openai_llm | parser

res = chain.invoke({"input": "计算机"})

print(res)

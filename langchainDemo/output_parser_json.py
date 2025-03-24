from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from utils import llm


class Joke(BaseModel):
    setup: str = Field(description="设置笑话的问题")
    res: str = Field(description="解决笑话的答案")


req = "告诉我一个笑话"

parser = JsonOutputParser(pydantic_object=Joke)

prompt = PromptTemplate(
    template="回答用户的查询. \n{setup}\n{res}\n",
    input_variables="res",
    partial_variables={"setup": parser.get_format_instructions()},
)

chain = prompt | llm | parser

res = chain.invoke({"res": req})

print(res)
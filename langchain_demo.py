from langchain.llms import Deepseek
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 配置Deepseek模型
llm = Deepseek(api_key="sk-09411ecfe7f940cfaf6bddc87b9fa00a", model="DeepSeek-V3")

# 创建Langchain的链
chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template("your_prompt_template"))

# 调用链
result = chain.run("your_input")
print(result)
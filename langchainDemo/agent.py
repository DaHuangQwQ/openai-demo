from langchainDemo.embedding import retriever_tool
from utils import llm
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor

tools = [retriever_tool]

prompt = hub.pull("hwchase17/openai-functions-agent")
print(prompt)

agent = create_tool_calling_agent(llm, tools, prompt)

agent_exec = AgentExecutor(agent=agent, tools=tools)

res = agent_exec.invoke({"input": "猫的特征？"})

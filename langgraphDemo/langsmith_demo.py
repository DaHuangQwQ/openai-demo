import asyncio
import operator
from typing import TypedDict, List, Annotated, Tuple, Union, Literal

from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import START
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from langchainDemo.utils import llm

# 搜索 api 库
tools = [TavilySearchResults(max_results=1)]

# prompt = hub.pull("wfh/react-agent-executor")
# prompt.pretty_print()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("placeholder", "{messages}"),
    ]
)

prompt.pretty_print()

agent_executor = create_react_agent(llm, tools, prompt=prompt)


# 用于存储输入、计划、过去的步骤和响应
class PlanExecute(TypedDict):
    input: str
    # 每个计划
    plan: List[str]
    # 步骤执行的情况
    past_steps: Annotated[List[Tuple], operator.add]
    res: str


# 用于描述未来要执行的计划
class Plan(BaseModel):
    """未来要执行的计划"""

    steps: List[str] = Field(description="需要执行的不同步骤，应该按顺序排列")


# 创建一个计划生成的提示模版
plan_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """对于给定的目标，提出一个简单的逐步计划。
            这个计划应该包含独立的任务，如果正确执行讲得出正确的答案。
            不要添加任何多余的步骤。
            最后一步的结果应该是最终答案。
            确保每一步都有所有必要的信息 - 不要跳过步骤。""",
        ),
        ("placeholder", "{messages}"),
    ]
)

# 创建一个计划生成器，结构化输出
planner = plan_prompt | llm.with_structured_output(Plan)


class Res(BaseModel):
    """用户响应"""

    res: str


# 用户描述执行的行为
class Act(BaseModel):
    """要执行的行为"""

    action: Union[Res, Plan] = Field(
        description="要执行的行为。如果要回应用户，使用Res。如果需要进一步使用工具获取答案，使用Plan。"
    )


# 重新计划的提示次模版
replanner_prompt = ChatPromptTemplate.from_template("""
对于给定的目标，提出一个简单的逐步计划。
这个计划应该包含独立的任务，如果正确执行讲得出正确的答案。
不要添加任何多余的步骤。
最后一步的结果应该是最终答案。
确保每一步都有所有必要的信息 - 不要跳过步骤。

你的目标是：
{input}

你的原计划是：
{plan}

你目前已完成的步骤是：
{past_steps}

相应地更新你的计划。
如果不需要更多步骤并且可以返回给用户，那么就这样回应。
如果需要，填写计划。
只添加仍然需要完成的步骤。不要返回己完成的步骤作为计划的一部分。
""")

replanner = replanner_prompt | llm.with_structured_output(Act)


async def main():
    # 生成计划步骤
    async def plan_step(state: PlanExecute):
        plan = await planner.ainvoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}

    # 用于执行步骤
    async def execute_step(state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_format = f"""
对于以下计划：
{plan_str}\n\n你的任务是执行第{1}步，{task}。
"""
        agent_res = await agent_executor.ainvoke({"messages": [("user", task_format)]})
        return {
            "past_steps": state["past_steps"]
            + [(task, agent_res["messages"][-1].content)]
        }

    # 用于重新计划步骤
    async def replan_step(state: PlanExecute):
        res = await replanner.ainvoke(state)
        if isinstance(res.action, Res):
            return {"res": res.action.res}
        else:
            return {"plan": res.action.steps}

    # 用于判断是否结束
    def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
        if "res" in state and state["res"]:
            return "__end__"
        else:
            return "agent"

    #
    workflow = StateGraph(PlanExecute)

    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_edge("agent", "replan")
    workflow.add_conditional_edges("replan", should_end)

    app = workflow.compile()
    graph_image = app.get_graph().draw_mermaid_png()
    with open("langsmith_demo.png", "wb") as f:
        f.write(graph_image)

    config = {"recursion_limit": 50}

    inputs = {"input": "2021年东京奥运会100米自由泳决赛冠军的家乡是哪里?请用中文答复"}

    async for event in app.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


# 运行异步函数
asyncio.run(main())

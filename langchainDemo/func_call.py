from typing import Literal

from utils import llm
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool


@tool
def weather_tool(weather: Literal["晴朗的", "多云的", "多雨的", "下雪的"]) -> None:
    """
    获取天气信息
    """
    pass


llm_with_tool = llm.bind_tools([weather_tool])

msg = HumanMessage(
    content="今天的天气怎么样",
)

res = llm_with_tool.invoke([msg])

print(res)

from langchain_core.tools import tool

@tool
def mul(a: int, b: int):
    """
    俩个数相乘
    """
    return a * b


print(mul.name)
print(mul.description)
print(mul.args)
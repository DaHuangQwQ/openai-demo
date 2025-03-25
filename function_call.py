import json
import requests
import os
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

GPT_MODEL = "gpt-3.5-turbo"


# 使用了retry库，指定在请求失败时的重试策略。
# 这里设定的是指数等待（wait_random_exponential），时间间隔的最大值为40秒，并且最多重试3次（stop_after_attempt(3)）。
# 定义一个函数chat_completion_request，主要用于发送 聊天补全 请求到OpenAI服务器
@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(
    messages, functions=None, function_call=None, model=GPT_MODEL
):
    # 设定请求的header信息，包括 API_KEY
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY"),
    }

    # 设定请求的JSON数据，包括GPT 模型名和要进行补全的消息
    json_data = {"model": model, "messages": messages}

    # 如果传入了functions，将其加入到json_data中
    if functions is not None:
        json_data.update({"functions": functions})

    # 如果传入了function_call，将其加入到json_data中
    if function_call is not None:
        json_data.update({"function_call": function_call})

    # 尝试发送POST请求到OpenAI服务器的chat/completions接口
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        # 返回服务器的响应
        return response

    # 如果发送请求或处理响应时出现异常，打印异常信息并返回
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


# 定义一个函数pretty_print_conversation，用于打印消息对话内容
def pretty_print_conversation(messages):
    # 为不同角色设置不同的颜色
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }

    # 遍历消息列表
    for message in messages:
        # 如果消息的角色是"system"，则用红色打印“content”
        if message["role"] == "system":
            print(
                colored(
                    f"system: {message['content']}\n", role_to_color[message["role"]]
                )
            )

        # 如果消息的角色是"user"，则用绿色打印“content”
        elif message["role"] == "user":
            print(
                colored(f"user: {message['content']}\n", role_to_color[message["role"]])
            )

        # 如果消息的角色是"assistant"，并且消息中包含"function_call"，则用蓝色打印"function_call"
        elif message["role"] == "assistant" and message.get("function_call"):
            print(
                colored(
                    f"assistant[function_call]: {message['function_call']}\n",
                    role_to_color[message["role"]],
                )
            )

        # 如果消息的角色是"assistant"，但是消息中不包含"function_call"，则用蓝色打印“content”
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(
                colored(
                    f"assistant[content]: {message['content']}\n",
                    role_to_color[message["role"]],
                )
            )

        # 如果消息的角色是"function"，则用品红色打印“function”
        elif message["role"] == "function":
            print(
                colored(
                    f"function ({message['name']}): {message['content']}\n",
                    role_to_color[message["role"]],
                )
            )


# 定义一个名为functions的列表，其中包含两个字典，这两个字典分别定义了两个功能的相关参数

# 第一个字典定义了一个名为"get_current_weather"的功能
functions = [
    {
        "name": "get_current_weather",  # 功能的名称
        "description": "Get the current weather",  # 功能的描述
        "parameters": {  # 定义该功能需要的参数
            "type": "object",
            "properties": {  # 参数的属性
                "location": {  # 地点参数
                    "type": "string",  # 参数类型为字符串
                    "description": "The city and state, e.g. San Francisco, CA",  # 参数的描述
                },
                "format": {  # 温度单位参数
                    "type": "string",  # 参数类型为字符串
                    "enum": ["celsius", "fahrenheit"],  # 参数的取值范围
                    "description": "The temperature unit to use. Infer this from the users location.",  # 参数的描述
                },
            },
            "required": ["location", "format"],  # 该功能需要的必要参数
        },
    },
    # 第二个字典定义了一个名为"get_n_day_weather_forecast"的功能
    {
        "name": "get_n_day_weather_forecast",  # 功能的名称
        "description": "Get an N-day weather forecast",  # 功能的描述
        "parameters": {  # 定义该功能需要的参数
            "type": "object",
            "properties": {  # 参数的属性
                "location": {  # 地点参数
                    "type": "string",  # 参数类型为字符串
                    "description": "The city and state, e.g. San Francisco, CA",  # 参数的描述
                },
                "format": {  # 温度单位参数
                    "type": "string",  # 参数类型为字符串
                    "enum": ["celsius", "fahrenheit"],  # 参数的取值范围
                    "description": "The temperature unit to use. Infer this from the users location.",  # 参数的描述
                },
                "num_days": {  # 预测天数参数
                    "type": "integer",  # 参数类型为整数
                    "description": "The number of days to forecast",  # 参数的描述
                },
            },
            "required": ["location", "format", "num_days"],  # 该功能需要的必要参数
        },
    },
]


# 定义一个空列表messages，用于存储聊天的内容
messages = []

# 使用append方法向messages列表添加一条系统角色的消息
messages.append(
    {
        "role": "system",  # 消息的角色是"system"
        "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",  # 消息的内容
    }
)

# 向messages列表添加一条用户角色的消息
messages.append(
    {
        "role": "user",  # 消息的角色是"user"
        "content": "What's the weather like today",  # 用户询问今天的天气情况
    }
)

# 使用定义的chat_completion_request函数发起一个请求，传入messages和functions作为参数
chat_response = chat_completion_request(messages, functions=functions)

# 解析返回的JSON数据，获取助手的回复消息
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的回复消息添加到messages列表中
messages.append(assistant_message)

pretty_print_conversation(messages)

type(assistant_message)

# 向messages列表添加一条用户角色的消息，用户告知他们在苏格兰的格拉斯哥
messages.append(
    {
        "role": "user",  # 消息的角色是"user"
        "content": "I'm in Shanghai, China.",  # 用户的消息内容
    }
)

# 再次使用定义的chat_completion_request函数发起一个请求，传入更新后的messages和functions作为参数
chat_response = chat_completion_request(messages, functions=functions)

# 解析返回的JSON数据，获取助手的新的回复消息
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的新的回复消息添加到messages列表中
messages.append(assistant_message)

pretty_print_conversation(messages)

# 初始化一个空的messages列表
messages = []

# 向messages列表添加一条系统角色的消息，要求不做关于函数参数值的假设，如果用户的请求模糊，应该寻求澄清
messages.append(
    {
        "role": "system",  # 消息的角色是"system"
        "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
    }
)

# 向messages列表添加一条用户角色的消息，用户询问在未来x天内苏格兰格拉斯哥的天气情况
messages.append(
    {
        "role": "user",  # 消息的角色是"user"
        "content": "what is the weather going to be like in Shanghai, China over the next x days",
    }
)

# 使用定义的chat_completion_request函数发起一个请求，传入messages和functions作为参数
chat_response = chat_completion_request(messages, functions=functions)

# 解析返回的JSON数据，获取助手的回复消息
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的回复消息添加到messages列表中
messages.append(assistant_message)

# 打印助手的回复消息
pretty_print_conversation(messages)

# 向messages列表添加一条用户角色的消息，用户指定接下来的天数为5天
messages.append(
    {
        "role": "user",  # 消息的角色是"user"
        "content": "5 days",
    }
)

# 使用定义的chat_completion_request函数发起一个请求，传入messages和functions作为参数
chat_response = chat_completion_request(messages, functions=functions)

# 解析返回的JSON数据，获取第一个选项
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的回复消息添加到messages列表中
messages.append(assistant_message)

# 打印助手的回复消息
pretty_print_conversation(messages)

# 在这个代码单元中，我们强制GPT 模型使用get_n_day_weather_forecast函数
messages = []  # 创建一个空的消息列表

# 添加系统角色的消息
messages.append(
    {
        "role": "system",  # 角色为系统
        "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
    }
)

# 添加用户角色的消息
messages.append(
    {
        "role": "user",  # 角色为用户
        "content": "Give me a weather report for San Diego, USA.",
    }
)

# 使用定义的chat_completion_request函数发起一个请求，传入messages、functions以及特定的function_call作为参数
chat_response = chat_completion_request(
    messages, functions=functions, function_call={"name": "get_n_day_weather_forecast"}
)

# 解析返回的JSON数据，获取第一个选项
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的回复消息添加到messages列表中
messages.append(assistant_message)

# 打印助手的回复消息
pretty_print_conversation(messages)

# 如果我们不强制GPT 模型使用 get_n_day_weather_forecast，它可能不会使用
messages = []  # 创建一个空的消息列表

# 添加系统角色的消息
messages.append(
    {
        "role": "system",  # 角色为系统
        "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
    }
)

# 添加用户角色的消息
messages.append(
    {
        "role": "user",  # 角色为用户
        "content": "Give me a weather report for San Diego, USA.",
    }
)

# 使用定义的chat_completion_request函数发起一个请求，传入messages和functions作为参数
chat_response = chat_completion_request(messages, functions=functions)

# 解析返回的JSON数据，获取第一个选项
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的回复消息添加到messages列表中
messages.append(assistant_message)

# 打印助手的回复消息
pretty_print_conversation(messages)

# 创建另一个空的消息列表
messages = []

# 添加系统角色的消息
messages.append(
    {
        "role": "system",  # 角色为系统
        "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.",
    }
)

# 添加用户角色的消息
messages.append(
    {
        "role": "user",  # 角色为用户
        "content": "Give me the current weather (use Celcius) for Toronto, Canada.",
    }
)

# 使用定义的chat_completion_request函数发起一个请求，传入messages、functions和function_call作为参数
chat_response = chat_completion_request(
    messages, functions=functions, function_call="none"
)

# 解析返回的JSON数据，获取第一个选项
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的回复消息添加到messages列表中
messages.append(assistant_message)

# 打印助手的回复消息
pretty_print_conversation(messages)

### 定义一个执行SQL查询的函数


import sqlite3

conn = sqlite3.connect("data/chinook.db")
print("Opened database successfully")


def get_table_names(conn):
    """返回一个包含所有表名的列表"""
    table_names = []  # 创建一个空的表名列表
    # 执行SQL查询，获取数据库中所有表的名字
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    # 遍历查询结果，并将每个表名添加到列表中
    for table in tables.fetchall():
        table_names.append(table[0])
    return table_names  # 返回表名列表


def get_column_names(conn, table_name):
    """返回一个给定表的所有列名的列表"""
    column_names = []  # 创建一个空的列名列表
    # 执行SQL查询，获取表的所有列的信息
    columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    # 遍历查询结果，并将每个列名添加到列表中
    for col in columns:
        column_names.append(col[1])
    return column_names  # 返回列名列表


def get_database_info(conn):
    """返回一个字典列表，每个字典包含一个表的名字和列信息"""
    table_dicts = []  # 创建一个空的字典列表
    # 遍历数据库中的所有表
    for table_name in get_table_names(conn):
        columns_names = get_column_names(conn, table_name)  # 获取当前表的所有列名
        # 将表名和列名信息作为一个字典添加到列表中
        table_dicts.append({"table_name": table_name, "column_names": columns_names})
    return table_dicts  # 返回字典列表


# 获取数据库信息，并存储为字典列表
database_schema_dict = get_database_info(conn)

# 将数据库信息转换为字符串格式，方便后续使用
database_schema_string = "\n".join(
    [
        f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
        for table in database_schema_dict
    ]
)

# 定义一个功能列表，其中包含一个功能字典，该字典定义了一个名为"ask_database"的功能，用于回答用户关于音乐的问题
functions = [
    {
        "name": "ask_database",
        "description": "Use this function to answer user questions about music. Output should be a fully formed SQL query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            SQL query extracting info to answer the user's question.
                            SQL should be written using this database schema:
                            {database_schema_string}
                            The query should be returned in plain text, not in JSON.
                            """,
                }
            },
            "required": ["query"],
        },
    }
]


def ask_database(conn, query):
    """使用 query 来查询 SQLite 数据库的函数。"""
    try:
        results = str(conn.execute(query).fetchall())  # 执行查询，并将结果转换为字符串
    except Exception as e:  # 如果查询失败，捕获异常并返回错误信息
        results = f"query failed with error: {e}"
    return results  # 返回查询结果


def execute_function_call(message):
    """执行函数调用"""
    # 判断功能调用的名称是否为 "ask_database"
    if message["function_call"]["name"] == "ask_database":
        # 如果是，则获取功能调用的参数，这里是 SQL 查询
        query = json.loads(message["function_call"]["arguments"])["query"]
        # 使用 ask_database 函数执行查询，并获取结果
        results = ask_database(conn, query)
    else:
        # 如果功能调用的名称不是 "ask_database"，则返回错误信息
        results = f"Error: function {message['function_call']['name']} does not exist"
    return results  # 返回结果


# 创建一个空的消息列表
messages = []

# 向消息列表中添加一个系统角色的消息，内容是 "Answer user questions by generating SQL queries against the Chinook Music Database."
messages.append(
    {
        "role": "system",
        "content": "Answer user questions by generating SQL queries against the Chinook Music Database.",
    }
)

# 向消息列表中添加一个用户角色的消息，内容是 "Hi, who are the top 5 artists by number of tracks?"
messages.append(
    {"role": "user", "content": "Hi, who are the top 5 artists by number of tracks?"}
)

# 使用 chat_completion_request 函数获取聊天响应
chat_response = chat_completion_request(messages, functions)

# 从聊天响应中获取助手的消息
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的消息添加到消息列表中
messages.append(assistant_message)

# 如果助手的消息中有功能调用
if assistant_message.get("function_call"):
    # 使用 execute_function_call 函数执行功能调用，并获取结果
    results = execute_function_call(assistant_message)
    # 将功能的结果作为一个功能角色的消息添加到消息列表中
    messages.append(
        {
            "role": "function",
            "name": assistant_message["function_call"]["name"],
            "content": results,
        }
    )

# 使用 pretty_print_conversation 函数打印对话
pretty_print_conversation(messages)


# 向消息列表中添加一个用户的问题，内容是 "What is the name of the album with the most tracks?"
messages.append(
    {"role": "user", "content": "What is the name of the album with the most tracks?"}
)

# 使用 chat_completion_request 函数获取聊天响应
chat_response = chat_completion_request(messages, functions)

# 从聊天响应中获取助手的消息
assistant_message = chat_response.json()["choices"][0]["message"]

# 将助手的消息添加到消息列表中
messages.append(assistant_message)

# 如果助手的消息中有功能调用
if assistant_message.get("function_call"):
    # 使用 execute_function_call 函数执行功能调用，并获取结果
    results = execute_function_call(assistant_message)
    # 将功能的结果作为一个功能角色的消息添加到消息列表中
    messages.append(
        {
            "role": "function",
            "content": results,
            "name": assistant_message["function_call"]["name"],
        }
    )

# 使用 pretty_print_conversation 函数打印对话
pretty_print_conversation(messages)

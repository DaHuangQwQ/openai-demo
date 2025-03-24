import base64

from langchain_deepseek import ChatDeepSeek

from dotenv import load_dotenv

load_dotenv()

# 配置Deepseek模型
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


def img_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
    base64_encoded = base64.b64encode(image_data)
    base64_string = base64_encoded.decode("utf-8")
    return base64_string

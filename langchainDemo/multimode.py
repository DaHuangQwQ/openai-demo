from utils import llm, img_to_base64
from langchain_core.messages import HumanMessage

image_data = img_to_base64("./_image.jpg")

msg = HumanMessage(
    content=[
        {"type": "text", "text": "请你描述一下图中的女子"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ]
)

res = llm.invoke([msg])

print(res.content)

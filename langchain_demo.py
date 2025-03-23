from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import Chroma

from langchain_deepseek.chat_models import ChatDeepSeek
from langchain_openai.embeddings import OpenAIEmbeddings

# 聊天场景的提示次模版
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

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

# few-shot
examples = [
    {
        "question": "What is your name?",
        "answer": "<NAME>",
    }
]

example_template = PromptTemplate(input_variables=['question'], output_variables=['answer'], template="问题：{question}\\n{answer}")

# embedding
# 利用 寓意相似度 做匹配
# example_selector = SemanticSimilarityExampleSelector.from_examples(
#     examples,
#     OpenAIEmbeddings(),
#     # 相似性搜索，开源的 向量数据库
#     Chroma,
#     k=1,
# )

few_shot_prompt = FewShotPromptTemplate(
    # example_selector=example_selector,
    examples=examples,
    example_prompt=example_template,
    suffix="问题: {input}",
    input_variables=["input"]
)

print(few_shot_prompt.format(input="你是谁"))

# template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是世界级的专家"),
        ("human", "{input}")
    ]
)

output_parser = StrOutputParser()

# chain
chain = prompt | llm | output_parser

# res = chain.invoke({"input": "说出10个关于程序员的笑话"})

# print(res)
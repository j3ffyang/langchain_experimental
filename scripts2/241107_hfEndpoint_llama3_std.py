
from langchain_core.prompts import PromptTemplate
question = "What is NFT?"
template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)


import os
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

from langchain_huggingface import HuggingFaceEndpoint
# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# repo_id = "google/gemma-2b"
repo_id = "meta-llama/Llama-3.2-1B"
llm = HuggingFaceEndpoint(
    repo_id = repo_id,
    temperature = 0.5,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    max_new_tokens = 250,
)

chain = prompt | llm
print(chain.invoke({"question": question}))

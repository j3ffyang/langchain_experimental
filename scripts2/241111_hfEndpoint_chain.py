import os
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

from langchain_huggingface import HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id = repo_id,
    temperature = 0.5,
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN,
    max_new_tokens = 250,
)

from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables = ["city"],
    template = "Describe a perfect day in {city}?"
)

chain = prompt | llm
print(chain.invoke("cairo"))

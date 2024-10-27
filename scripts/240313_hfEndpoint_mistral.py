# https://docs.kanaries.net/articles/langchain-chains-what-is-langchain

from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["city"],
    template="Describe a perfect day in {city}?")


# from langchain_community.llms import HuggingFaceHub
# llm = HuggingFaceEndpoint(repo_id="EleutherAI/gpt-neo-2.7B")
from langchain_community.llms import HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# repo_id = "EleutherAI/gpt-neo-2.7B"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_new_tokens=1024, temperature=0.5)

from langchain.chains import LLMChain
llmchain = LLMChain(llm=llm, prompt=prompt, verbose=True)
print(llmchain.invoke("Paris"))

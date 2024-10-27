# https://docs.kanaries.net/articles/langchain-chains-what-is-langchain

from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["city"],
    template="Describe a perfect day in {city}?")


from langchain.llms import HuggingFaceHub
llm = HuggingFaceHub(repo_id="EleutherAI/gpt-neo-2.7B")
# llm = HuggingFaceHub(repo_id="google/flan-t5-small",
#                      model_kwargs={
#                          "temperature": 0,
#                          "max_length": 64
#                      }
#                     )


from langchain.chains import LLMChain
from langchain.globals import set_debug, set_verbose
set_debug(True)
set_verbose(True)
llmchain = LLMChain(llm=llm, prompt=prompt, verbose=True)
print(llmchain.run("Paris"))

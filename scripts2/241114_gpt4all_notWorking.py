
from langchain_core.prompts import PromptTemplate
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)


from langchain_core.callbacks import BaseCallbackHandler
count = 0

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        global count
        if count < 10:
            print(f"Token: {token}")
            count += 1

# from langchain_community.llms import gpt4all
# local_path = (
#     "/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf"
# )
# 
# # llm = GPT4All(model=local_path, callbacks=[MyCustomHandler()], streaming=True)
# llm = gpt4all(model=local_path, n_threads=8)

from gpt4all import GPT4All
llm = GPT4All("/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf")


chain = prompt | llm

question = "What is NFT?"

res = chain.invoke({"question": question})

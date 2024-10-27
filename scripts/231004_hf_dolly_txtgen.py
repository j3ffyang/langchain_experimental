# https://www.packtpub.com/article-hub/making-the-best-out-of-hugging-face-hub-using-langchain

import pprint
from langchain_community.llms import HuggingFaceHub

repo_id = "databricks/dolly-v2-3b"
llm = HuggingFaceHub(repo_id=repo_id,
                     model_kwargs={"temperature": 5, "max_length": 100})

from langchain.prompts import PromptTemplate
template = """Question: {question}

Answer: Let's think step by step"""
prompt = PromptTemplate(template=template, input_variables=["question"])

from langchain.chains import LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "In the first movie of Harry Potter, what is the name of three-headed dog?"
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(llm_chain.invoke(question))

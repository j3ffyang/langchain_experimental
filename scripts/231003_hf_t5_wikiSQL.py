# https://www.youtube.com/watch?v=dD_xNmePdd0

from pprint import pprint
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

hub_llm = HuggingFaceHub(repo_id="mrm8488/t5-base-finetuned-wikiSQL")

prompt = PromptTemplate(
    input_variables=["question"],
    template="Translate English to SQL: {question}"
)

hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
# pprint(hub_chain.run("What is the average age of the respondents using a mobile device?"))
# pprint(hub_chain.run("What is the median  age of the respondents using a mobile device?"))
query = input("Please input your query: ")
pprint(hub_chain(query))

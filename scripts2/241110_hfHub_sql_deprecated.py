from pprint import pprint

from langchain_community.llms import HuggingFaceHub
llm = HuggingFaceHub(repo_id="mrm8488/t5-base-finetuned-wikiSQL")

from langchain.prompts import PromptTemplate
prompt = PromptTemplate(
    input_variables=["question"],
    template="Translate English to SQL: {question}"
)

from langchain_core.runnables import RunnableLambda
chain = prompt | llm

pprint(chain.invoke("What is the average age of the respondents using a mobile device?"))
pprint(chain.invoke("What is the median  age of the respondents using a mobile device?"))

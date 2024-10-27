# https://stackoverflow.com/questions/78330865/problem-with-lcel-lanchais-rag-sequential-chains

from source.main.create_db import vectordb 
from langchain.llms import OpenAI
from langchain import prompts
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain import memory
from dotenv import load_dotenv
import os as os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document


load_dotenv()
key = os.getenv('OPENAI_API_KEY')
llm = OpenAI()
retriever = vectordb.as_retriever()

template = '''
ou are an assistant specialized in Brazilian public exams, 
mainly focused on tests in the field of law,
and must answer the questions below
{question}
'''

prompt= ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()
model = llm

chain1 = prompt | model | output_parser 


template2 = '''
Correct and enhance the {answer} using your new information from the 
{context} and provide more references and support for the newly crafted response."
'''

prompt2 =  ChatPromptTemplate.from_template(template2)
setup_and_retrieval = RunnableParallel(
    {'context':retriever , "answer":RunnablePassthrough()}
)

chain2 = ({'answer': chain1} | setup_and_retrieval | prompt2 | model | StrOutputParser() )


a = chain2.invoke('tell me one law')

print(a)


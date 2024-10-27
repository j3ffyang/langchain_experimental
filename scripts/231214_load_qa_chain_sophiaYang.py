# https://github.com/sophiamyang/tutorials-LangChain/blob/main/LangChain_QA.ipynb
# sophiayang

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('/home/jeff/Downloads/231214_linux_refdoc.pdf')
documents = loader.load()

from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
chain = load_qa_chain(llm=OpenAI(temperature=0.5), chain_type="map_reduce")

question = "what is talking about in this book?"
print(chain.run(input_documents=documents, question=question))

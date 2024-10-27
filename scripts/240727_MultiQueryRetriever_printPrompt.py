# https://stackoverflow.com/questions/78800797/how-to-view-the-final-prompt-in-a-multiqueryretriever-pipeline-using-langchain
# https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/MultiQueryRetriever/
# cp ~/Downloads/240102_Hans-Christian-Andersen-Fairy-Tales_short.pdf /tmp/

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
# loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")
loader = DirectoryLoader('/tmp/', glob="./*.pdf")
data = loader.load()
print(data)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# vectordb = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory='/tmp/chromadb')
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
# vertordb.persist()


from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.llms import Ollama
llm = Ollama(model="mistral-nemo", base_url="http://127.0.0.1:11434")


import logging

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


from typing import List

from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from a vector
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vectordb.as_retriever(), llm, prompt=QUERY_PROMPT
)

template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""


from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(template)


from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from langchain.globals import set_verbose, set_debug

set_debug(True)
set_verbose(True)


def inspect(state):
    """Print the state passed between Runnables in a langchain and pass it on"""
    print(state)
    return state


qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RunnableLambda(inspect)  # Add the inspector here to print the intermediate results
    | prompt
    | llm
    | StrOutputParser()
)

# Invoke the QA chain with a sample query
qa_chain.invoke("Give 10 quotes from this articles related to love?")


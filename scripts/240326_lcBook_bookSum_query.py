## Define embedding model
from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


## Configure Qdrant client, and define vectorstore and retriever
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
client = QdrantClient("127.0.0.1", port="6333")
vectorstore = Qdrant(client, embeddings=embedding, collection_name="worldhistory")
retriever = vectorstore.as_retriever()


## Define prompt and prompttemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
template = """You are an assistant for question-answering tasks. Use the
following pieces of retrieved context to answer the question. If you don't know
the answer, just say that you don't know. Use three sentences maximum and keep
the answer concise.
Question: {question}
Context: {context}
Answer:
"""
# prompt = ChatPromptTemplate.from_template(template)
prompt = PromptTemplate.from_template(template)


## Define LLM
from langchain_community.llms import Ollama
# llm = Ollama(model="gemma:2b")
llm = Ollama(model="mistral")


## Create an RAG chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


while True:
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    else:
        print(rag_chain.invoke(user_input))

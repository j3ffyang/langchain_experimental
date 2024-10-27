## prereq > cp ~/Downloads/240326_WorldHistory.pdf /tmp/ 

## Load PDF documents from a directory
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
loader = DirectoryLoader('/tmp/', glob="**/*.pdf", loader_cls=PyPDFLoader,
                         show_progress=True, use_multithreading=True)
## Sample code for loading doc from web
# from langchain_community.document_loaders import WebBaseLoader
# loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")
documents = loader.load()


## Split the documents into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)


## Define embedding model
from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


## Configure Qdrant client, and define vectorstore and retriever
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
url = "http://127.0.0.1:6333"
vectorstore = Qdrant.from_documents(
    docs,
    embedding=embedding,
    url=url,
    # prefer_grpc=True,
    collection_name="worldhistory",
)
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

# query = "What is this story telling about?"
# print(rag_chain.invoke(query))

while True:
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    else:
        print(rag_chain.invoke(user_input))


# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# loader = DirectoryLoader('/tmp/', glob="**/*.pdf", loader_cls=PyPDFLoader,
#                         show_progress=True, use_multithreading=True)
# documents = loader.load()
# 
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# docs = splitter.split_documents(documents)

from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    # model_name = "sentense-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_name = "sentence-transformers/all-mpnet-base-v2"
)

from langchain_qdrant import QdrantVectorStore
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding = embedding,
    url = "http://127.0.0.1:6333",
    collection_name = "worldHist",
)
retriever = vectorstore.as_retriever()

from langchain.prompts import ChatPromptTemplate
template = """You are an assistant for question-answering tasks. Use the
following pieces of retrieved context to answer the question. If you don't know
the answer, just say that you don't know. Use three sentences maximum and keep
the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)


import os
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# from langchain_huggingface import HuggingFaceEndpoint
# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# llm = HuggingFaceEndpoint(
#     repo_id = repo_id,
#     temperature = 0.5,
#     huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
#     max_new_tokens = 250,
# )
from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="mistral",
    temperature=0.5,
)

from langchain.globals import set_verbose, set_debug
set_debug(True)
set_verbose(True)

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


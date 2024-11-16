from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")
document = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(document)

from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

from langchain_community.vectorstores import FAISS
vectorstore= FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()

from langchain.prompts import ChatPromptTemplate
template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. Translate the answer into psychological and medical explanation as a humor. Pretend that I am a 10 year old boy
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

from langchain_huggingface import HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id = repo_id, max_new_tokens = 250, temperature = 0.5
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

query = "What is this story telling about in Chinese language?"
print(rag_chain.invoke(query))

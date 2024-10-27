from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI


loader = DirectoryLoader('/tmp/', glob="./*.pdf")
document = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(document)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # model_name="bert-base-multilingual-cased")

vectordb = Chroma.from_documents(chunks, embedding, persist_directory="/tmp/chromadb")
vectordb.persist()


chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.5),
        retriever=vectordb.as_retriever(), chain_type="stuff",
        # chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True)

response = chain.invoke("please summarize this book")
print(response)

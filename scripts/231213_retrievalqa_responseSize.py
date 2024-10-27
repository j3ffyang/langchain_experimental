# https://stackoverflow.com/questions/77651318/how-to-increase-the-response-size-of-chromadb

from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# def loadFiles():

# loader = DirectoryLoader('/tmp/', glob="./*.pdf", loader_cls=PyPDFLoader)
loader = DirectoryLoader('/tmp/', glob="./*.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=100)
texts = text_splitter.split_documents(documents)
# return texts


# def createDb(load,embeddings,persist_directory):
# max_input_size = 3000
# num_output = 256
# chunk_size_limit = 1000 # token window size per document
# max_chunk_overlap = 80 # overlap for each token fragment

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

persist_directory = "/tmp/chromadb"
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
vectordb.persist()
# return vectordb


qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0, model_name="text-davinci-003"),
        retriever=vectordb.as_retriever(), chain_type="stuff",
        # chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True)

response = qa_chain("please summarize this book")
print(response)

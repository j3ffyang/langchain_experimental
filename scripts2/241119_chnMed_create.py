
file_path = "/home/jeff/Downloads/scratch/instguid.git/hlmGPT/chineseMed2/data"
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader
loader = DirectoryLoader("/home/jeff/Downloads/scratch/instguid.git/hlmGPT/data/chn-med/",
                         glob="**/*.docx",
                         loader_cls=Docx2txtLoader,
                         show_progress=True)
documents = loader.load()


from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(documents)

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="" 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-m3"
# model_kwargs = {"device": "cuda"}
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embedding = HuggingFaceBgeEmbeddings(
    model_name = model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


from langchain_qdrant import QdrantVectorStore
url = "http://127.0.0.1:6333"
vectorstore = QdrantVectorStore.from_documents(
    docs,
    embedding,
    url = url,
    collection_name = "chnMed",
)

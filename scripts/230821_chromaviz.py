from chromaviz import visualize_collection
from langchain.vectorstores import Chroma


data
vectordb = Chroma.from_documents("/home/jeff/Downloads/scratch/instguid.git/hlmGPT/tutorials/scripts/chroma_db")
visualize_collection(vectordb, "lang_docs.html")
# https://stackoverflow.com/questions/78287802/document-search-with-azure-openai-and-faiss-is-not-working/78290010#78290010

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("/tmp/240102_Hans-Christian-Andersen-Fairy-Tales_short.pdf")
pages = loader.load_and_split()
# pages = loader.load()
# print(pages)

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

from langchain_community.vectorstores import FAISS
faiss_index = FAISS.from_documents(pages, embeddings)

print(faiss_index)

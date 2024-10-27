# https://stackoverflow.com/questions/78256936/similarity-search-with-chromadb-and-default-setting-returns-4-identical-document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

loader = PyPDFLoader("/tmp/240326_WorldHistory.pdf")
pages = loader.load()
pages = pages[9:360]
text = ""
for page in pages:
    text += page.page_content
text = text.replace('\t', ' ')
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "\t"],
    chunk_size=1000, chunk_overlap=100)
docs = text_splitter.create_documents([text])

DB = Chroma.from_documents(
    docs,
    SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
    persist_directory="/tmp/chroma_db"
)

print(DB.similarity_search('china'))

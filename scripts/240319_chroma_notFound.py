# https://stackoverflow.com/questions/78187507/error-in-chroma-vector-database-langchain


from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")
documents = loader.load()


from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                               chunk_overlap=200)
texts = splitter.split_documents(documents)

# from langchain_openai import OpenAIEmbeddings
# embedding = OpenAIEmbeddings()

from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

from langchain_community.vectorstores import Chroma
db = Chroma.from_documents(texts, embedding)

print(type(db))

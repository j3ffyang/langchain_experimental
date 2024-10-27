# https://stackoverflow.com/questions/78222275/semantic-chunking-with-langchain-on-faiss-vectorstore

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
vectorstore = FAISS.from_texts(
    ["harry potter's owl is in the castle."], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()

with open("/tmp/240325_hotelCalifornia_reddit.txt") as f:
    docs = f.read()

print(docs)

from langchain_experimental.text_splitter import SemanticChunker
text_splitter = SemanticChunker(OpenAIEmbeddings())
docs = text_splitter.create_documents([docs])
print(docs)

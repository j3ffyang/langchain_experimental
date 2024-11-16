# from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Text_file")
document = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = splitter.split_documents(document)

from langchain_huggingface import HuggingFaceEmbeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs,
)

from langchain_community.vectorstores import Qdrant
vectorstore = Qdrant.from_documents(
    docs,
    embedding = embedding,
    location = ":memory:",
    collection_name = "wikipedia",
)

query = "What's flatfile?"
# result = vectorstore.similarity_search_with_score(query, k=2)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k":2},
)

# print(retriever.get_relevant_documents(query)[0])
print(retriever.invoke(query))

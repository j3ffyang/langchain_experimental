# langchain book > loader > ask a question

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Text_file")

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = loader.load_and_split(text_splitter)

# documents = loader.load()
# docs = text_splitter.split_documents(documents)

from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings

# vectorstore = Qdrant(embeddings=HuggingFaceEmbeddings(), url="127.0.0.1:6333", collection_name="disney")
vectorstore = Qdrant.from_documents(
    chunks,
    # docs,
    embedding=HuggingFaceEmbeddings(),
    # url="127.0.0.1:6333",
    location=":memory:",
    collection_name="wikipedia",
    )


query = "What's flatfile?"
# results = vectorstore.similarity_search_with_score(query, k=2)
# print(results)

retriever = vectorstore.as_retriever()
# print(retriever)
print(retriever.get_relevant_documents(query)[0])


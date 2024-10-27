# https://python.langchain.com/docs/expression_language/cookbook/retrieval

# from langchain.chat_models import ChatOpenAI
# model = ChatOpenAI()

from langchain.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

from langchain.vectorstores import FAISS
vectorstore = FAISS.from_texts(
    ["harry potter's owl is in the castle"],
    embedding = embedding,
    )

print(vectorstore)

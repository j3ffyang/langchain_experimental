from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts(
    ["harry potter's owl is in the castle."],
    embedding = embedding,
)

print(vectorstore)

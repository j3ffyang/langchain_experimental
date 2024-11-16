# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    # model_name = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

text = "This is a test document."
query_result = embedding.embed_query(text)

print(query_result[:3])
print(type(query_result))

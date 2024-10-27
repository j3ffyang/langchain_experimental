from langchain.embeddings import HuggingFaceEmbeddings

# embeddings = HuggingFaceEmbeddings()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
text = "This is a test document."
query_result = embeddings.embed_query(text)
print(query_result[:3])
print(type(query_result))


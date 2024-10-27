# https://stackoverflow.com/questions/77948428/issue-with-qdrant-collection-creation-not-sure-about-the-format-which-input-fo

data = {
'name': ['Entry 1', 'Entry 2', 'Entry 3'],
'urls': ['http://example.com/1', 'http://example.com/2', 'http://example.com/3'],
'text': ['Text for Entry 1', 'Text for Entry 2', 'Text for Entry 3'],
'type': ['Type A', 'Type B', 'Type C']}

print(type(data))
print(data)

data2 = list(data.items())
print(type(data2))
print(data2)


from langchain_community.vectorstores import Qdrant
# texts = data["text"].tolist()
# model_name = "sentence-transformers/sentence-t5-base"
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name=model_name)

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

doc_store = Qdrant.from_texts(
    ["data"],
    embeddings,
    # url=qdrant_url,
    # api_key=qdrant_key,
    location=":memory:",
    collection_name="my-collection",
)

print(doc_store)

# https://stackoverflow.com/questions/77645107/chroma-db-not-working-in-both-persistent-and-http-client-modes/77647471?noredirect=1#comment136946759_77647471

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

vectorstore = Chroma.from_texts(["harry potter's owl is in the castle."], 
                        embedding_function, persist_directory="./chroma_db")
query = "where is Harry Potter's Hedwig?"
docs = vectorstore.similarity_search(query)
print(docs[0].page_content)

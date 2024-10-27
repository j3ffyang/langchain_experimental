# https://stackoverflow.com/questions/78836094/langchain-chat-history/78836389#78836389
# https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/

from langchain_ollama.llms import OllamaLLM
llm = OllamaLLM(model="tinyllama", base_url="http://127.0.0.1:11434")
# from langchain_community.llms import Ollama
# llm = Ollama(model="tinyllama", base_url="http://127.0.0.1:11434")


# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multiligual-MiniLM-L12-v2")
# print(embedding)


import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embedding_size = 384
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = embedding().embedd_query
vectorstore = FAISS(embedding_fn, index, InMemoryDoctore({}), {})

retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)

memory.save_context({"input": "My favorite food is pizza"}, {"output": "that's good to know"})
memory.save_context({"input": "My favorite sport is soccer"}, {"output": "..."})
memory.save_context({"input": "I don't the Celtics"}, {"output": "ok"}) #


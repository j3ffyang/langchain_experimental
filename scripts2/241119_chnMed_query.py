from langchain_community.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-base-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embedding = HuggingFaceBgeEmbeddings(
    model_name = model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)


from langchain_qdrant import QdrantVectorStore
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding = embedding,
    url = "http://127.0.0.1:6333",
    collection_name = "chnMed",
)
retriever = vectorstore.as_retriever()


import os
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# from langchain_huggingface import HuggingFaceEndpoint
# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# llm = HuggingFaceEndpoint(
#     repo_id = repo_id,
#     temperature = 0.5,
#     huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
#     max_new_tokens = 250,
# )
from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="qwen2.5",
    temperature=0.5,
)


from langchain.prompts import ChatPromptTemplate
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# from langchain.globals import set_verbose, set_debug
# set_debug(True)
# set_verbose(True)


from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# print(rag_chain)

# response = rag_chain.invoke({"input": "What does the emperor new cloth look like?"})
# print(response["answer"])


while True: 
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    else:
        print(rag_chain.invoke({"input": user_input})["answer"])

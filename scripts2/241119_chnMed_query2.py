from langchain_community.embeddings import HuggingFaceBgeEmbeddings
model_name = "BAAI/bge-m3"
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


from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="qwen2.5",
    temperature=0.5,
)


from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.chains.history_aware_retriever import create_history_aware_retriever

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

from langchain.globals import set_debug, set_verbose
set_debug(True)
set_verbose(True)

from langchain_core.messages import AIMessage, HumanMessage

chat_history = []

def process_question(question):
    ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
    chat_history.extend(
       [
           HumanMessage(content=question),
           AIMessage(content=ai_msg["answer"]),
       ]
    )
    return ai_msg["answer"]


from pprint import pprint
while True:
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    else:
        pprint(process_question(user_input))

import bs4
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    # web_paths=("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes",),
    web_paths=("https://andersen.sdu.dk/vaerk/hersholt/TheEmperorsNewClothes_e.html",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)


from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


import os
import getpass
# os.environ["SUPABASE_URL"] = getpass.getpass("Supabase URL: ")
# os.environ["SUPABASE_SERVICE_KEY"] = getpass.getpass("Supabase Service Key: ")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
# supabase_url = os.environ.get("SUPABASE_URL")
# supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase_url = SUPABASE_URL
supabase_key = SUPABASE_SERVICE_KEY
supabase_client = create_client(supabase_url, supabase_key)

# # create a new collection
# vectorstore = SupabaseVectorStore.from_documents(
#     chunks,
#     embedding = embedding,
#     client = supabase,
#     table_name = "documents",
#     query_name = "match_documents",
#     chunk_size = 500,
# )

# query an existing collection
vectorstore = SupabaseVectorStore(
    embedding = embedding,
    client = supabase_client,
    table_name = "documents",
    query_name = "match_documents",
)

## from langchain_community.vectorstores import Qdrant
## vectorstore = Qdrant.from_documents(
##     documents,
##     embedding = embedding,
##     location = ":memory:",
##     collection_name = "temp",
## )

# retriever = vectorstore.as_retriever(search_type = "mmr")
retriever = vectorstore.as_retriever()


from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="mistral",
    temperature=0.5,
)

import os
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# from langchain_huggingface import HuggingFaceEndpoint
# # repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# # repo_id = "google/gemma-2b"
# repo_id = "meta-llama/Llama-3.2-1B"
# llm = HuggingFaceEndpoint(
#     repo_id = repo_id,
#     temperature = 0.5,
#     huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
#     max_new_tokens = 250,
# )


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

# from langchain_core.messages import AIMessage, HumanMessage

chat_history = []

question = "What does the emperor like?"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
print(ai_msg_1)

# chat_history.extend(
#     [
#         HumanMessage(content=question),
#         AIMessage(content=ai_msg_1["answer"]),
#     ]
# )
# 
# second_question = "Who is he?"
# ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})
# 
# print(ai_msg_2)
# 
# 
# # from pprint import pprint
# # while True:
# #     user_input = input("Enter your question: ")
# #     if user_input == "exit":
# #         break
# #     else:
# #         pprint(rag_chain.invoke({"input": user_input, "chat_history": chat_history}))

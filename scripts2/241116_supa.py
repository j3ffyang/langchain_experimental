import bs4
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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
supabase: Client = create_client(supabase_url, supabase_key)

vectorstore = SupabaseVectorStore.from_documents(
    chunks,
    embedding, 
    client = supabase,
    table_name = "documents",
    query_name = "match_document",
    chunk_size = 500,
)


from langchain_ollama import ChatOllama
llm = ChatOllama(
    model="mistral",
    temperature=0.5,
)


# from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

prompt = ChatPromptTemplate(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)


from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


from langchain.chains.llm import LLMChain
legacy_chain = LLMChain(
    llm = llm,
    prompt=prompt,
    memory=memory,
)

legacy_result = legacy_chain.invoke({"text": "my name is bob"})
print(legacy_result)

legacy_result = legacy_chain.invoke({"text": "what was my name"})

# from pprint import pprint
# while True:
#     user_input = input("Enter your question: ")
#     if user_input == "exit":
#         break
#     else:
#         pprint(legacy_chain.invoke({"text": user_input}))

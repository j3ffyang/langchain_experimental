# https://python.langchain.com/docs/modules/memory/adding_memory_chain_multiple_inputs/
# https://www.linode.com/docs/guides/installing-supabase/


from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


import os
# import getpass
# os.environ["SUPABASE_URL"] = getpass.getpass("Supabase URL: ")
# os.environ["SUPABASE_SERVICE_KEY"] = getpass.getpass("Supabase Svc Key: ")


from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
# supabase_url = os.environ.get("SUPABASE_URL")
# supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase_url = "http://127.0.0.1:8000"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyAgCiAgICAicm9sZSI6ICJhbm9uIiwKICAgICJpc3MiOiAic3VwYWJhc2UtZGVtbyIsCiAgICAiaWF0IjogMTY0MTc2OTIwMCwKICAgICJleHAiOiAxNzk5NTM1NjAwCn0.dc_X5iR_VP_qT0zsiyj_I_OZ2T9FtRU2BBNWN8Bu4GE"
supabase: Client = create_client(supabase_url, supabase_key)

vectorstore = SupabaseVectorStore(
    # chunks,
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    # chunk_size=500,
)


from langchain_core.prompts import PromptTemplate
template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document and a question, create a final answer.

{context}

{chat_history}
Human :{human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"],
    template=template
)


from langchain_community.llms import Ollama
llm = Ollama(model="mistral")
# llm = Ollama(model="llama3")
# llm = Ollama(model="gemma:2b")


from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff", memory=memory, prompt=prompt)


from pprint import pprint
while True:
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    else:
        retriever = vectorstore.similarity_search(user_input)
        pprint(chain.invoke({"input_documents": retriever, "human_input": user_input}, return_only_outputs=True))
        # print(chain.memory.buffer)

# query = "What does the emperor say"
# chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
# pprint(chain.memory.buffer)


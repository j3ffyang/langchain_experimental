# https://python.langchain.com/docs/modules/memory/adding_memory_chain_multiple_inputs/

from pprint import pprint

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)


from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(
    # chunks, embeddings, metadatas=[{"source": i} for i in range(len(chunks))]
    chunks, embeddings
)
retriever = vectorstore.as_retriever(k=4)
# query = "What does the emperor say?"
# docs = vectorstore.similarity_search(query)
# pprint(docs)


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


from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff", memory=memory, prompt=prompt)

while True:
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    else:
        docs = vectorstore.similarity_search(user_input)
        pprint(chain.invoke({"input_documents": docs, "human_input": user_input}, return_only_outputs=True))
        # print(chain.memory.buffer)

# query = "What does the emperor say"
# chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
# pprint(chain.memory.buffer)


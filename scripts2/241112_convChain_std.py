# https://python.langchain.com/docs/tutorials/qa_chat_history/#agents

import bs4
import os
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    web_paths=("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

from langchain_huggingface import HuggingFaceEmbeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embedding = HuggingFaceEmbeddings(
    model_name = model_name, model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs,
)

# from langchain_core.vectorstores import InMemoryVectorStore
# vectorstore = InMemoryVectorStore.from_documents(
#     documents=splits, embedding=embedding,
# )
from langchain_community.vectorstores import Qdrant
vectorstore = Qdrant.from_documents(
    docs,
    embedding = embedding,
    location = ":memory:",
    collection_name = "temp",
)
retriever = vectorstore.as_retriever()


from langchain_huggingface import HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id = repo_id,
    temperature = 0.5,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    max_new_tokens = 250,
)

from langchain_core.prompts import ChatPromptTemplate
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


from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# print(rag_chain)

response = rag_chain.invoke({"input": "What does the emperor new cloth look like?"})
print(response["answer"])

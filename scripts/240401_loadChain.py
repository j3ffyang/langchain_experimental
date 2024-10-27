# https://stackoverflow.com/questions/78256263/how-to-load-the-retrievalqa-model-from-a-file

import os
os.environ["http_proxy"] = "http://127.0.0.1:8889"
os.environ["https_proxy"]= "http://127.0.0.1:8889"

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")
# from langchain_community.document_loaders import TextLoader, DirectoryLoader
# loader = DirectoryLoader('/tmp/', glob="./*.pdf")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
documents = splitter.split_documents(data)


from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

from langchain_community.vectorstores.chroma import Chroma
vectordb = Chroma.from_documents(documents=documents, embedding=embedding,
                                 persist_directory="/tmp/chromadb")
vectordb.persist()
retriever = vectordb.as_retriever()


from langchain.prompts import PromptTemplate
# from langchain.prompts import ChatPromptTemplate
template = """You are an assistant for question-answering tasks. Use the
following pieces of retrieved context to answer the question. If you don't know
the answer, just say that you don't know. Use three sentences maximum and keep
the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["context", "question"])
# prompt = ChatPromptTemplate.from_template(template)


from langchain_community.llms import HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_new_tokens=250, temperature=0.5)


# from langchain_openai.llms import OpenAI
# llm = OpenAI()


from langchain.globals import set_verbose, set_debug
set_debug(True)
set_verbose(True)

from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
)

# print(qa_chain.invoke("what happened to the emperor?"))

chain.save(file_path="/tmp/chain_model.yaml")

# from langchain.chains import load_chain
# chain = load_chain(
#     path="/tmp/chain_model.yaml",
#     retriever = retriever,
#     # model = llm,
# )

print(chain.invoke("what happened to the emperor?"))

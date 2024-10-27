import shutil
import os
shutil.copy("/home/jeff/Downloads/249192_Hans-Christian-Andersen-Fairy-Tales-1.pdf", "/tmp")

# from langchain.document_loaders import WebBaseLoader
# loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")
from langchain_community.document_loaders import TextLoader, DirectoryLoader
loader = DirectoryLoader('/tmp/', glob="./*.pdf")
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                               chunk_overlap=200)
texts = text_splitter.split_documents(documents)


from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

from langchain_community.vectorstores import Chroma
persist_directory = "/tmp/chromadb"
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings,
                                 persist_directory=persist_directory)
vectordb.persist()

from langchain.prompts import PromptTemplate
# prompt_template = """
# Compare the book given in question with others in the retriever based on genre and description.
# Return a complete sentence with the full title of the book and describe the similarities between the books.
# 
# question: {question}
# context: {context}
# """
prompt_template = """Write a concise summary of the following: "{context}" CONCISE SUMMARY: """
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# https://python.langchain.com/docs/integrations/providers/gpt4all
# https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-bindings/python
from langchain_community.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# Instantiate the model. Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(
    model="/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf",
    device='gpu',
    n_threads=8)


from langchain.globals import set_verbose, set_debug
set_debug(True)
set_verbose(True)

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(), chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True)

from langchain.chains import RetrievalQA
response = qa_chain("please summarize this book")
# response = qa_chain("what happened to the emperor?")

import pprint
pp = pprint.PrettyPrinter(indent=0)
# pp.pprint(response)
pp.pprint(response['result'])
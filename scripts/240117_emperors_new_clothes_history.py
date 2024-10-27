import os
os.environ["http_proxy"] = "http://127.0.0.1:8889"
os.environ["https_proxy"] = "http://127.0.0.1:8889"

import argparse
parser = argparse.ArgumentParser(description='Usage\n cp ~/Downloads/240102_Hans-Christian-Andersen-Fairy-Tales_short.pdf /tmp')
# parser = argparse.ArgumentParser(description='Usage\n cp
#                                  ~/Downloads/240102_Hans-Christian-Andersen-Fairy-Tales_short.pdf
#                                  /tmp/\n rm -fr /tmp/chromadb; py
#                                  231231_emperors_new_clothes.py')
args = parser.parse_args()

from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)


from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")
# from langchain.document_loaders import TextLoader, DirectoryLoader
# loader = DirectoryLoader('/tmp/', glob="./*.pdf")
texts = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                               chunk_overlap=200)
documents = text_splitter.split_documents(texts)


from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # model_name="bert-base-multilingual-cased")


from langchain_community.vectorstores import Chroma
persist_directory = "/tmp/chromadb"
vectordb = Chroma.from_documents(documents=documents, embedding=embedding,
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


# from langchain.llms import OpenAI
# llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")

# from langchain.llms import HuggingFaceHub
# repo_id = "Writer/camel-5b-hf"
# repo_id = "Salesforce/xgen-7b-8k-base"
# repo_id = "lmsys/fastchat-t5-3b-v1.0"
# repo_id = "google/flan-t5-base"
# llm = HuggingFaceHub(repo_id = "google/flan-t5-base",
#                      model_kwargs={"temperature":0.6,"max_length": 500, "max_new_tokens": 200})
# llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64})


# from langchain.llms.huggingface_pipeline import HuggingFacePipeline
# llm = HuggingFacePipeline.from_model_id(
#       model_id="microsoft/DialoGPT-medium", task="text-generation", pipeline_kwargs={"max_new_tokens": 200, "pad_token_id": 50256},)

# llm = HuggingFacePipeline.from_model_id(
#     # model_id="gpt2",  # 胡說
#     # model_id="lmsys/fastchat-t5-3b-v1.0",
#     model_id="bigscience/bloom-1b7",    # the working one
#     task="text-generation",
#     # device=0,
#     # batch_size=2,
#     pipeline_kwargs={"max_new_tokens": 200, "pad_token_id": 50256})


# https://python.langchain.com/docs/integrations/providers/gpt4all
# https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-bindings/python
from langchain_community.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# Instantiate the model. Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(
    model="/home/jeff/.cache/gpt4all/mistral-7b-openorca.Q4_0.gguf",
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

# from langchain.chains import RetrievalQA
# response = qa_chain("what is a quark?")
response = qa_chain("what happened to the emperor?")

import pprint
pp = pprint.PrettyPrinter(indent=0)
# pp.pprint(response)
pp.pprint(response['result'])


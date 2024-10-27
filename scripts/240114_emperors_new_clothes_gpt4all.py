import argparse
parser = argparse.ArgumentParser(description='Usage\n cp ~/Downloads/240102_Hans-Christian-Andersen-Fairy-Tales_short.pdf /tmp')
# parser = argparse.ArgumentParser(description='Usage\n cp
#                   ~/Downloads/240102_Hans-Christian-Andersen-Fairy-Tales_short.pdf
#                   /tmp/\n rm -fr /tmp/chromadb; py
#                   231231_emperors_new_clothes.py')
args = parser.parse_args()

from warnings import filterwarnings
filterwarnings("ignore", category=FutureWarning)


from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")
# from langchain_community.document_loaders import TextLoader, DirectoryLoader
# loader = DirectoryLoader('/tmp/', glob="./*.pdf")
document = loader.load()


from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunks = splitter.split_documents(document)


from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # model_name="bert-base-multilingual-cased")

from langchain_community.vectorstores import Chroma
vectordb = Chroma.from_documents(documents=chunks, embedding=embedding,
                                 persist_directory="/tmp/chromadb")
vectordb.persist()

from langchain.prompts import PromptTemplate
# prompt_template = """
# Compare the book given in question with others in the retriever based on genre and description.
# Return a complete sentence with the full title of the book and describe the similarities between the books.
# 
# question: {question}
# context: {context}
# """
template = """Write a concise summary of the following: "{context}" CONCISE SUMMARY: """
prompt = PromptTemplate(template=template, input_variables=["context", "question"])


# from langchain.llms import OpenAI
# llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")

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
    retriever=vectordb.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)
response = qa_chain("please summarize this book")
# response = qa_chain("what happened to the emperor?")

import pprint
pp = pprint.PrettyPrinter(indent=0)
# pp.pprint(response)
pp.pprint(response['result'])

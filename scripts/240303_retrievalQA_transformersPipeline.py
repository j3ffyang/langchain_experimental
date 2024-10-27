# https://huggingface.co/distilbert/distilgpt2?library=true

# from langchain_community.document_loaders import WebBaseLoader
# loader = WebBaseLoader("https://en.wikipedia.org/wiki/Text_file")
# document = loader.load()
# 
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
# docs = splitter.split_documents(document)


from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts(
    ["harry potter's owl is in the castle."],
    embedding = embedding,
)


from langchain.prompts import PromptTemplate
PROMPT = PromptTemplate.from_template(
    """
    Use the following pieces of context to answer the question at the end. If
    you don't know the answer, just say that you don't know, don't try to make
    up an answer.

    {context}

    Question: {question}
    Helpful Answer:
    """
)


# from transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
# model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

from transformers import pipeline
pipeline = pipeline(
    "text-generation",
    model="distilbert/distilgpt2",
    # model = "google-bert/bert-base-chinese",
    max_new_tokens=256,
    # max_length=100
)
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0.5})


from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever
retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=VectorStoreRetriever(vectorstore=vectorstore),
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    callbacks=[StdOutCallbackHandler()],
)


import pprint
pp = pprint.PrettyPrinter(indent=0)
response = retrievalQA.invoke("where is harry potter's owl?")
pp.pprint(response)

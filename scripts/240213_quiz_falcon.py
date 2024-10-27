# https://stackoverflow.com/questions/77975106/create-quizzes-from-pdfs-with-llama-2-and-langchain/77975324#77975324

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Pinecone
# import pinecone
# from huggingface_hub import notebook_login
import os
import sys


#  Creating a Llama2 Model Wrapper
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# model_id = "tiiuae/falcon-7b-instruct"
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)
pipeline = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    # trust_remote_code=True,
    device_map="auto",
    max_new_tokens=100,
    # max_length=200,
)

# notebook_login()
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_auth_token=True)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
#                                          device_map='auto',
#                                          torch_dtype=torch.float16,
#                                          use_auth_token=True,
#                                          load_in_8bit=True
#                                          )
# pipe = pipeline("text-generation",
#             model=model,
#             tokenizer= tokenizer,
#             torch_dtype=torch.bfloat16,
#             device_map="auto",
#             max_new_tokens = 512,
#             do_sample=True,
#             top_k=30,
#             num_return_sequences=1,
#             eos_token_id=tokenizer.eos_token_id
#             )


from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipeline)
# temperature value is from 0 to 1, the high values means the model is more creative
# llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature':0.5})


from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
examples = [
    {"question": "What is the capital of France?",
     "answers": ["Paris*", "London", "Berlin", "Madrid"]},
    {"question": "Who is the author of 'To Kill a Mockingbird'?",
     "answers": ["Harper Lee*", "J.K. Rowling", "Stephen King", "Agatha Christie"]},
    {"question": "What is the largest planet in our solar system?",
     "answers": ["Jupiter*", "Mars", "Saturn", "Venus"]},
    {"question": "What is the tallest mountain in the world?",
        "answers": ["Mount Everest*", "K2", "Kangchenjunga", "Lhotse"]},
    {"question": "What is the capital of the United States?",
        "answers": ["Washington D.C.*", "New York", "Los Angeles", "Chicago"]},
    {"question": "What is the smallest country in the world?",
        "answers": ["Vatican City*", "Monaco", "Nauru", "Tuvalu"]},
    {"question": "What is the largest ocean in the world?",
        "answers": ["Pacific Ocean*", "Atlantic Ocean", "Indian Ocean", "Arctic Ocean"]},
]

example_prompt = PromptTemplate(
    input_variables=["question", "answer"], template="Question: {question}\n{answer}"
)

print(example_prompt)
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}",
    input_variables=["input"],
)


from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # model_name="bert-base-multilingual-cased")


# from langchain.vectorstores import Chroma
# persist_directory = "/tmp/chromadb"
# docsearch = Chroma.from_documents(documents=texts, embedding=embeddings,
#                                  persist_directory=persist_directory)
# docsearch.persist()

from langchain_community.vectorstores import FAISS
# vectorstore = FAISS.load_local("vectorstore/db_faiss_bkb", embeddings)
vectorstore = FAISS.from_texts(["harry potter's owl is in the castle"],
                               embedding = embeddings)
# retriever = vectorstore.as_retriever(search_kwargs={'k': 1, 'score_treshold': 0}, search_type="similarity")


from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

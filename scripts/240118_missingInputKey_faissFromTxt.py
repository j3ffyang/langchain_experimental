# https://stackoverflow.com/questions/77839844/langchain-retrievalqa-missing-some-input-keys
# DOES NOT WORK
# Comment> 
# RunnablePassthrough function is alternative of RetrievalQA in LangChain. This
# function ensures to set variables, like query, for both prompt and retriever.
# In your previous code, the variables got set in retriever, but not in prompt.
# That's why LLM complains the missing keys.

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",
#     model_kwargs={'device': 'mps'}, encode_kwargs={'device': 'mps', 'batch_size': 32})

from langchain.vectorstores import FAISS
# vectorstore = FAISS.load_local("vectorstore/db_faiss_bkb", embeddings)
vectorstore = FAISS.from_texts(["harry potter's owl is in the castle"],
                               embedding = embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 1, 'score_treshold': 0}, search_type="similarity")

# llm = build_llm("modelle/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf")
from langchain_community.llms import GPT4All
llm = GPT4All(model="/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf", n_threads=8)


from langchain.prompts import PromptTemplate
# prompt = """
prompt_template = """
<s> [INST] You are getting a task and a User input. If there is relevant information in the context, please add this information to your answer.

### Here the context: ###
{context}

### Here the User Input: ###
{query}

Answer: [/INST]
"""

# ### Here the Task: ###
# {typescript_string}

# prompt_temp = PromptTemplate(template=prompt, input_variables=["typescript_string", "context", "query"])
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])

from langchain.chains import RetrievalQA
# def build_retrieval_qa(llm, prompt, vectordb):
#     dbqa = RetrievalQA.from_chain_type(llm=llm,
#                                         chain_type='stuff',
#                                         retriever=vectordb,
#                                         return_source_documents=True,
#                                         chain_type_kwargs={"prompt": prompt},
#                                         verbose=True)
#     return dbqa
dbqa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type='stuff',
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
    verbose=True)


from langchain.globals import set_verbose, set_debug
set_debug(True)
set_verbose(True)

question = "What is IGB?"
# types = "Answer shortly!"
context = "What is IGB?"

# dbqa1 = build_retrieval_qa(llm=llm,prompt=prompt_temp,vectordb=retriever)
# dbqa1 = dbqa({"query": question, "context": context, "typescript_string": types})
dbqa1 = dbqa({"question": question, "context": context})

print(dbqa1)

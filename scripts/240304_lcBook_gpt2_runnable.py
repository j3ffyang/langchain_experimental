# from langchain_community.embeddings import HuggingFaceEmbeddings
# embedding = HuggingFaceEmbeddings(
#     model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# )

# from langchain_community.vectorstores import FAISS
# vectorstore = FAISS.from_texts(
#     ["harry potter's owl is in the castle."],
#     embedding = embedding,
# )

# from langchain.prompts import PromptTemplate
# prompt = PromptTemplate.from_template(
#     """
#     Use the following pieces of context to answer the question at the end. If
#     you don't know the answer, just say that you don't know, don't try to make
#     up an answere.
# 
#     {context}
# 
#     Question: {question}
#     Helpful Answer:
#     """
# )


from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

from transformers import pipeline
pipeline = pipeline(
    "text-generation",
    model="distilbert/distilgpt2",
    max_new_tokens=200,
    # max_length=100
)
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0.5})


from langchain.schema.output_parser import StrOutputParser
from langchain.callbacks.tracers import ConsoleCallbackHandler

from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser
chain.invoke({"topic": "UK"}, config={'callbacks': [ConsoleCallbackHandler()]})

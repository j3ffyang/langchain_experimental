# https://stackoverflow.com/questions/78176373/i-keep-getting-the-same-error-when-using-huggingfacepipeline/78178100#78178100

from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
# from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
# from langchain import PromptTemplate, HuggingFaceHub, LLMChain
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
# import os

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "my api"
model_id = "google/flan-t5-xxl"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
# 
# pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=128)
# local_llm = HuggingFacePipeline(pipeline=pipeline)

from langchain_community.llms import HuggingFaceEndpoint
local_llm = HuggingFaceEndpoint(
    repo_id=model_id, max_new_tokens=128, temperature=0.5)

prompt = PromptTemplate(
    input_variables=["name"],
    template="Can you tell me about the politician {name}"
)

chain = LLMChain(llm=local_llm, prompt=prompt)
print(chain.invoke("Donald Trump"))

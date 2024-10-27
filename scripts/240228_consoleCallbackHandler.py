from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.callbacks.tracers import ConsoleCallbackHandler

# from langchain_community.llms import HuggingFaceHub
# llm = HuggingFaceHub(
#     repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
#     model_kwargs={'temperature': 0.9, 'max_length': 5555500}
# )

# from transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
# model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
# 
# from transformers import pipeline
# pipeline = pipeline(
#     "text-generation",
#     model="distilbert/distilgpt2",
#     max_new_tokens=100,
#     max_length=100
# )
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# llm = HuggingFacePipeline(pipeline=pipeline)

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
model = ChatOpenAI()
# model = llm()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

chain.invoke({"topic": "tintin"}, config={'callbacks': [ConsoleCallbackHandler()]})

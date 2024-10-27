# https://stackoverflow.com/questions/77868284/combining-falcon-40b-instruct-with-langchain

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


from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipeline)


from langchain.prompts import PromptTemplate
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | llm
question = "Tell me about Italy"

print(chain.invoke({"question": question}))

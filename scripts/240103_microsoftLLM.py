from langchain.llms.huggingface_pipeline import HuggingFacePipeline

hf = HuggingFacePipeline.from_model_id(
    model_id="microsoft/DialoGPT-medium", task="text-generation", device=0, pipeline_kwargs={"max_new_tokens": 200, "pad_token_id": 50256},
)

from langchain.prompts import PromptTemplate

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

question = "What is electroencephalography?"

print(chain.invoke({"question": question}))


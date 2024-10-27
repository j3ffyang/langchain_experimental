# https://stackoverflow.com/questions/77947395/langchain-hugging-face-huggingfacepipeline-error/77948715

# from transformers import pipeline
# import langchain
from langchain_community.llms import HuggingFacePipeline

model_name = "bert-base-uncased"
# task = "question-answering"
task = "text-generation"

# hf_pipeline = pipeline(task, model=model_name)

langchain_pipeline = HuggingFacePipeline.from_model_id(
    model_name,
    task,
    # is_decoder=True,
    # device=gpt,
)

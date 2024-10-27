# https://www.kdnuggets.com/2023/08/langchain-streamlit-llama-bringing-conversational-ai-local-machine.html

from llama_cpp import Llama
from langchain.llms import HuggingFaceHub

# llm = Llama(model_path="/home/jeff/.cache/huggingface/hub/models--llama-7b/llama-7b.ggmlv3.q4_0.bin")
llm = Llama(model_path="/home/jeff/.cache/huggingface/hub/models--distilgpt2/snapshots/38cc92ec43315abd5136313225e95acc5986876c/pytorch_model.bin")

response = llm("Who directed The Dark Knight?")

print(response['choices'][0]['text'])

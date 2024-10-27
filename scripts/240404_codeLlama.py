# https://huggingface.co/TheBloke/CodeLlama-70B-Python-GGUF

from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/CodeLlama-7B-Python-GGUF",
    model_file="/home/jeff/.cache/huggingface/hub/codellama-7b-python.Q4_0.gguf",
    model_type="llama",
    gpu_layers=50)

question = "write me a python code to display calendar"
print(llm(question))


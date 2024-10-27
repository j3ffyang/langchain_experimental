from langchain_community.llms import Ollama

# llm = Ollama(model="llama2")
llm = Ollama(model="mistral")

print(llm.invoke("Tell me a joke"))

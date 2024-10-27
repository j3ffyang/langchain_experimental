# https://stackoverflow.com/questions/78129399/python-langchain-openai-self-errors-multiple

# from langchain_openai import OpenAI
from langchain_community.llms import HuggingFaceEndpoint

def gen_pet_name():
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_new_tokens=1024, temperature=0.5)
    name = llm.invoke("I have a cat, and I want a cool name for my cat")
    return name

if __name__ == "__main__":
    print(gen_pet_name())

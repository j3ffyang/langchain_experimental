# https://stackoverflow.com/questions/78129399/python-langchain-openai-self-errors-multiple

from langchain_openai import OpenAI

def gen_pet_name():
    llm = OpenAI(temperature=0.7)
    name = llm("I have a cat, and I want a cool name for my cat")
    return name

if __name__ == "__main__":
    print(gen_pet_name())

# https://www.youtube.com/watch?v=dD_xNmePdd0

from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

hub_llm = HuggingFaceHub(
    repo_id="gpt2",
    model_kwargs={'temperature': 0.5, 'max_length': 200}
    )

prompt = PromptTemplate(
    template="What are the top {k} places to do {this} in 2023?",
    input_variables=["k", "this"]
)

hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)
print(hub_chain.invoke({"k":3, "this": "tour"}))

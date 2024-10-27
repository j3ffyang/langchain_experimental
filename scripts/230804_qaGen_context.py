import openai
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate

openai.api_key = os.getenv("OPENAI_API_KEY")

openai = OpenAI(
    model_name="text-davinci-003",
    openai_api_key = openai.api_key
)

template = """Instruction: Answer the questions based on the context below. If the question cannot be answered using the information provided answer with "I don't know".

Context: Large Language Models (LLMs) are the latest models used in NLP. Their superior performance over smaller models has made them incredibly usefult for developers building NLP enabled applications. These models can be accessed via Hugging Face's `transformers` library, via OpenAI using the `openai` library, and via Cohere using the `cohere` library.

Question: {query}

Answer: """

prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
)

# print(openai(prompt))
print(openai(
    prompt_template.format(
        query="Which libraries and model providers offer LLMs?"
    )
))
import openai
import os
from langchain.llms import OpenAI
from langchain import PromptTemplate

openai.api_key = os.getenv("OPENAI_API_KEY")

openai = OpenAI(
    model_name="text-davinci-003",
    openai_api_key = openai.api_key
)

prompt = """The following is a conversation with an AI assistant. The assistant is typically sarcastic and witty, producing create and funny responses to the users questions. Here are some examples.

User: What is the meaning of life?
AI: """

openai.temperature = 1.0    # increase creativity/ randomness of output

print(openai(prompt))
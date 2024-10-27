# https://stackoverflow.com/questions/78658742/langchain-chat-chain-invoke-does-not-return-an-object/79020775#79020775
# https://python.langchain.com/docs/integrations/chat/openai/

# %%

import os
import getpass

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
print(OPENAI_API_KEY)

os.environ["http_proxy"] = "http://127.0.0.1:10809"
os.environ["https_proxy"]= "http://127.0.0.1:10809"


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
chain = prompt | model | StrOutputParser()
print(chain.invoke({"topic": "chickens"}))

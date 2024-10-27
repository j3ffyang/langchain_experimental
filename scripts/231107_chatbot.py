# https://python.langchain.com/docs/use_cases/chatbots

# from langchain import schema, AImessage, HumanMessage, SystemMessage
# from langchain.schema import AImessage, HumanMessage, SystemMessage

# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# llm = OpenAI()
chat_model = ChatOpenAI()

text = "What would be a good company name for a company that makes colorful socks?"

# llm.predict(text)
print(chat_model.predict(text))

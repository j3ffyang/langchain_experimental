# https://python.langchain.com/docs/modules/memory/adding_memory

from langchain.chains import LLMChain
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

template = """You are a chatbot having a conversation with a human.

{chat_history}
Human: {human_input}
Chatbot: """

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
                     template=template
)
memory = ConversationBufferMemory(memory_key="chat_history")

llm = OpenAI()
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

llm_chain.predict(human_input="Hi there my friend")

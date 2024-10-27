# https://python.langchain.com/docs/use_cases/chatbots/memory_management/

from pprint import pprint

from langchain_community.llms import Ollama
llm = Ollama(model="mistral")


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

query = "Translate this sentence from English to Chinese: I Love programming."

from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
demo_ephemeral_chat_history_for_chain = ChatMessageHistory()
chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: demo_ephemeral_chat_history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# result = chain_with_message_history.invoke(
#     {"input": query},
#     {"configurable": {"session_id": "unused"}},
# )


while True:
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    else:
        result=chain_with_message_history.invoke(
            {"input": user_input},
            {"configurable": {"session_id": "unused"}},
        )
        print(result)

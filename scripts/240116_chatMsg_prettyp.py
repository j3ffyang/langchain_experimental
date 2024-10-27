# https://stackoverflow.com/questions/77828572/what-is-sent-to-the-llm-when-using-a-chat-model/77828779

from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

messages = template.format_messages(
    name="Bob",
    user_input="What is your name?"
)

# prompt = template.format(
#     name="Bob",
#     user_input="What is your name?"
# )

# print(prompt)

import pprint
pp = pprint.PrettyPrinter(indent=0)
pp.pprint(messages)

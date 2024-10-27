# https://stackoverflow.com/questions/77661834/chatmessagehistory-into-a-runnable-langchain

# Init ChatMessageHistory object
from langchain.memory import ChatMessageHistory
history = ChatMessageHistory()

messages = "what's the weather today?"

# # Loop thru each message in the list
# for message in messages:
#     if message['is_from'] == 'human':
#         history.add_user_message(message['message'])
#     else:
#         history.add_ai_message(message['message'])

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

{history}

Answer the question on based on the context provide below:
{context}

Here is a question:
{question}"""


chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough(),
                      "history": history})
    | RunnableLambda(prompt_router)
    | model
    | StrOutputParser()
)

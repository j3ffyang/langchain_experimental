# https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/

from langchain_community.llms import GPT4All
llm = GPT4All(
    model="/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf",
    device='gpu',
    n_threads=8)


from langchain.chains import ConversationChain
conversation = ConversationChain(llm=llm)
print(conversation.prompt)

# from langchain.chains.conversation.memory import ConversationBufferMemory
# conversation_buf = ConversationChain(
#     llm=llm,
#     memory=ConversationBufferMemory()
# )
# print(conversation_buf("Good morning AI!"))

from langchain.chains.conversation.memory import ConversationBufferWindowMemory
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=1)
)

while True:
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    else:
        print(conversation(user_input))

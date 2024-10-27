# https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/

from langchain.llms import OpenAI
from langchain.chains import ConversationChain

llm = OpenAI(
    temperature=0,
    model_name="text-davinci-003"
    )

conversation = ConversationChain(llm=llm)

print(conversation.prompt.template)


from langchain.memory import ConversationBufferMemory

conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

# print(conversation_buf("Good morning AI!\n"))


from langchain.callbacks import get_openai_callback

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result

count_tokens(
    conversation_buf,
    "My interest here is to explore the potential of integrating Large Language Models with external knowledge"
)

count_tokens(
    conversation_buf,
    "I just want to analyze the different possibilities. What can you think of"
)

count_tokens(
    conversation_buf,
    "Which data source types could be used to give context to the model?"
)

count_tokens(
    conversation_buf,
    "What is my aim again?"
)

# print(conversation_buf.memory.buffer)

### use ConversationChain

from langchain.chains.conversation.memory import ConversationSummaryMemory

conversation = ConversationChain(
    llm=llm,
    memory=ConversationSummaryMemory(llm=llm)
)

print(conversation.memory.prompt.template)
# print(ConversationSummaryMemory.m
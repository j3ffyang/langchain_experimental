# https://towardsdatascience.com/the-easiest-way-to-interact-with-language-models-4da158cfb5c5

from langchain.memory import ConversationBufferWindowMemory, CombinedMemory, ConversationSummaryMemory

conv_memory = ConversationBufferWindowMemory(
    memory_key="chat_history_lines",
    input_key="input",
    k=1
)


from langchain_community.llms import GPT4All
# llm = OpenAI(temperature=0)
llm = GPT4All(model="/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf", n_threads=8)

summary_memory = ConversationSummaryMemory(llm=llm, input_key="input")
# Combined
memory = CombinedMemory(memories=[conv_memory, summary_memory])


from langchain.prompts import PromptTemplate
_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Summary of conversation:
{history}
Current conversation:
{chat_history_lines}
Human: {input}
AI:"""
PROMPT = PromptTemplate(
    input_variables=["history", "input", "chat_history_lines"], template=_DEFAULT_TEMPLATE
)


from langchain.chains import ConversationChain
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory,
    prompt=PROMPT
)

response = conversation.run("Tell me about Italy")
print(response)

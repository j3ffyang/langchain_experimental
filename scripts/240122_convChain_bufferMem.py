# https://python.langchain.com/docs/modules/memory/types/buffer
# https://medium.com/@michael.j.hamilton/conversational-memory-with-langchain-82c25e23ec60

from langchain_community.llms import GPT4All
llm = GPT4All(
    model="/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf",
    device='gpu',
    n_threads=8)

# from langchain_openai import OpenAI
# llm = OpenAI(temperature=0)


from langchain.globals import set_verbose, set_debug
set_debug(True)
set_verbose(True)


from langchain.prompts import PromptTemplate
# _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
# 
# Relevant pieces of previous conversation:
# {history}
# 
# (You do not need to use these pieces of information if not relevant)
# 
# Current conversation:
# Human: {input}
# AI:"""

_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. If the AI does not know the answer to a question, it truthfully says it does not know.

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Answer: {input}
"""

PROMPT = PromptTemplate(
   input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)


from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
conversation = ConversationChain(
    llm=llm,
    prompt=PROMPT,
    # memory=ConversationBufferMemory(),
    memory=ConversationBufferWindowMemory(k=3),
    verbose=True)


while True:
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    else:
        # result=conversation.predict(input="Hello!")
        result=conversation(user_input)
        print(result)

# https://stackoverflow.com/questions/78222548/how-do-i-amend-this-langchain-script-so-it-only-outputs-the-ai-response-but-is-s

from langchain_community.llms import HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    # repo_id=repo_id, max_new_tokens=250, temperature=0.5)
    repo_id="google/flan-t5-xxl", max_length=500, max_new_tokens=250, temperature=0.5
)

# from langchain_community.llms import GPT4All
# llm = GPT4All(
#     model="/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf",
#     device='gpu',
#     n_threads=8)

# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI()

from langchain.prompts import PromptTemplate
# TEMPLATE = """You're a helpful assistant, aiming at solving the problem.
# 
# Relevant pieces of previous conversation:
# {history}
# 
# (You do not need to use these pieces of information if not relevant. Please
# answer in one single sentence)
# 
# Answer my question: {input}
# """
TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI Assistant:"""
PROMPT = PromptTemplate(
   input_variables=["history", "input"], template=TEMPLATE
)

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# print(conversaion.invoke(input="Hi there!"))
while True:
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    else:
        result = conversation.predict(input=user_input)
        # result = conversation.invoke(input=user_input)
        print(result)

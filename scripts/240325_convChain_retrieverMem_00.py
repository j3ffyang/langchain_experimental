# https://milvus.io/blog/conversational-memory-in-langchain.md

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


from langchain_community.vectorstores import FAISS
from langchain.memory import VectorStoreRetrieverMemory
vectorstore = FAISS.from_texts(
    ["Harry Potter's owl is in the castle"], embedding=embeddings)
# retriever = vectorstore.as_retriever(vectorstore, search_kwargs=dict(k=1))
retriever = vectorstore.as_retriever(search_kwargs=dict(k=2))
memory = VectorStoreRetrieverMemory(retriever=retriever)


from langchain_community.llms import GPT4All
llm = GPT4All(
    model="/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf",
    device='gpu',
    n_threads=8)


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
_DEFAULT_TEMPLATE = """You're a helpful assistant, aiming at solving the problem.

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Answer my question: {input}
"""
PROMPT = PromptTemplate(
   input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)


from langchain.chains import ConversationChain
conversation_with_summary = ConversationChain(
   llm=llm,
   prompt=PROMPT,
   memory=memory,
   verbose=True
)
# result = conversation_with_summary.predict(input="Where is the hedwig? Why is it in castle?")

while True:
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    else:
        result = conversation_with_summary.predict(input=user_input)
        print(result)


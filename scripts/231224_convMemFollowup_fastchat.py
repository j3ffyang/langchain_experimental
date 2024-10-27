# https://python.langchain.com/docs/modules/memory/types/vectorstore_retriever_memory

from datetime import datetime
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub, OpenAI
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate


import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS

embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings().embed_query
# embedding_fn = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").embed_query
vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})


retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
memory = VectorStoreRetrieverMemory(retriever=retriever)

memory.save_context({"input": "My favorite food is pizza"}, {"output": "that's good to know"})
memory.save_context({"input": "My favorite sport is soccer"}, {"output": "..."})
memory.save_context({"input": "I don't the Celtics"}, {"output": "ok"})

print(memory.load_memory_variables({"prompt": "what sport should i watch?"})["history"])

llm = HuggingFaceHub(repo_id = "lmsys/fastchat-t5-3b-v1.0",
                     model_kwargs={"temperature": 0.3, "max_length": 1000})
# 
# llm = HuggingFaceHub(repo_id = "google/flan-t5-base",
#                      model_kwargs={"temperature":0.6,"max_length": 500, "max_new_tokens": 200 })
# 
# llm = HuggingFaceHub(repo_id="EleutherAI/gpt-neo-2.7B")

# llm = OpenAI(temperature=0)


_DEFAULT_TEMPLATE = """The following is a friendly conversation between a human
and an AI. The AI is talkative and provides lots of specific details from its
context. If the AI does not know the answer to a question, it truthfully says
it does not know.

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""

PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
)
conversation_with_summary = ConversationChain(
    llm=llm,
    prompt=PROMPT,
    memory=memory,
    verbose=True
)
response = conversation_with_summary.predict(input="Hi, my name is Perry, what's up?")
print(response)

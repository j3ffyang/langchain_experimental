# https://python.langchain.com/docs/modules/memory/chat_messages/
# https://python.langchain.com/docs/expression_language/how_to/map

from langchain.memory import ChatMessageHistory
history = ChatMessageHistory()

history.add_user_message("hi!")
history.add_ai_message("what's up?")

print(history.messages)
print(type(history.messages))

# lst = history.messages
# 
# def convert(lst):
#    res_dict = {}
#    for i in range(0, len(lst), 2):
#        res_dict[lst[i]] = lst[i + 1]
#    return res_dict

# print(convert(lst))
# 
# 
from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts(
    ["harry potter's owl is in the castle"], embedding=embedding,
)
retriever = vectorstore.as_retriever(search_kwargs=dict(k=2))


from langchain_community.llms import GPT4All
model = GPT4All(
    model="/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf",
    device='gpu',
    n_threads=8)


from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough(),
                      "history": history})
    | RunnableLambda(prompt_router)
    | model
    | StrOutputParser()
)

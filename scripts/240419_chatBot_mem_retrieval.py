# https://python.langchain.com/docs/use_cases/chatbots/quickstart/

from pprint import pprint

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)


from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(k=4)
# response = retriever.invoke(query)


from langchain_community.llms import Ollama
llm = Ollama(model="mistral")
# llm = Ollama(model="llama3")


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
document_chain = create_stuff_documents_chain(llm, question_answering_prompt)


from langchain.memory import ChatMessageHistory
demo_ephemeral_chat_history = ChatMessageHistory()

# demo_ephemeral_chat_history.add_user_message(query)
# response2 = document_chain.invoke(
#     {
#         "messages": demo_ephemeral_chat_history.messages,
#         "context": response,
#     }
# )
# pprint(response2)


## Creating a retrieval chain
from typing import Dict
from langchain_core.runnables import RunnablePassthrough

def parse_retriever_input(params: Dict):
    return params["messages"][-1].content
# 
# retrieval_chain = RunnablePassthrough.assign(
#     context=parse_retriever_input | retriever,
# ).assign(
#     answer=document_chain,
# )
# response3 = retrieval_chain.invoke(
#     {
#         "messages": demo_ephemeral_chat_history.messages,
#     }
# )
# pprint(response3)


retrieval_chain_with_only_answer = (
    RunnablePassthrough.assign(
        context=parse_retriever_input | retriever,
    )
    | document_chain
)


while True:
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    else:
        response=retrieval_chain_with_only_answer.invoke(
            {
                "messages": demo_ephemeral_chat_history.messages,
            }
        )
        print(response)

# response4 = retrieval_chain_with_only_answer.invoke(
#     {
#         "messages": demo_ephemeral_chat_history.messages,
#     }
# )
# pprint(response4)



# # from langchain_core.messages import HumanMessage
# # result = llm.invoke(
# #     [
# #         HumanMessage(
# #             content="Translate this sentence from English to Chinese: I love programming"
# #         )
# #     ]
# # )
# 
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system", 
#             "You are a helpful assistant. Answer all questions to the best of your ability.",
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )
# chain = prompt | llm
# 

# from langchain_community.llms import HuggingFaceEndpoint
# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# llm = HuggingFaceEndpoint(repo_id=repo_id, max_new_tokens=1024, temperature=0.5)

from langchain_openai.embeddings import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts(
    ["harry potter's owl is in the castle."], embedding)

from langchain_community.llms import HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# repo_id = "google/flan-t5-xxl"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_new_tokens=512, temperature=0.5)


from langchain_core.prompts import PromptTemplate
template = (
    "Combine the chat history and follow up question into "
    "a standalone question. Chat History: {chat_history}"
    "Follow up question: {question}"
)
prompt = PromptTemplate.from_template(template)

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
conversation = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    memory=memory,
    verbose=True,
)

# query = input("Please input your query: ")
# print(conversation_chain.invoke(query))

while True:
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    else:
        # result=conversation.predict(input="Hello!")
        result=conversation.invoke(user_input)
        print(result)


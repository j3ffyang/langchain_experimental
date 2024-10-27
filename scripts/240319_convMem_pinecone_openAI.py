# https://stackoverflow.com/questions/78188956/chatbot-that-uses-only-the-information-in-the-retriever-and-nothing-more

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

from langchain_pinecone import PineconeVectorStore
from langchain.memory import VectorStoreRetrieverMemory
vectorstore = PineconeVectorStore.from_texts(
    ["Harry Potter's owl is in the castle."], embedding=embedding,
    index_name="langchain-test-index")
retriever = vectorstore.as_retriever(search_kwargs=dict(k=2))
memory = VectorStoreRetrieverMemory(retriever=retriever)

from langchain.prompts import PromptTemplate
TEMPLATE = """You're a helpful assistant, aiming at solving the problem.

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Answer my question: {input}
"""
PROMPT = PromptTemplate(
   input_variables=["history", "input"], template=TEMPLATE
)

from langchain.chains import ConversationChain
conversation_with_summary = ConversationChain(
    llm=llm,
    prompt=PROMPT,
    memory=memory,
    verbose=True
)

while True:
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    else:
        result = conversation_with_summary.predict(input=user_input)
        print(result)

# https://stackoverflow.com/questions/76269666/i-ran-into-an-error-when-i-try-to-use-youtubeloader-from-youtube-url/78241334#78241334

from langchain_community.document_loaders import YoutubeLoader
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ChatVectorDBChain, ConversationalRetrievalChain

# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
# os.environ["OPENAI_API_KEY"] = "apikey"
loader = YoutubeLoader.from_youtube_url(youtube_url="https://www.youtube.com/watch?v=7OPg-ksxZ4Y",add_video_info=True)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 20
)

documents = text_splitter.split_documents(documents)
#print(documents)

embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(documents=documents,embedding=embeddings)
retriever = vector_store.as_retriever()

system_template = """
Use the following context to answer the user's question.
If you don't know the answer, say you don't, don't try to make it up. And answer in Chinese.
-----------
{context}
-----------
{chat_history}
"""

messages  =[
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template('{question}')
]

prompt = ChatPromptTemplate.from_messages(messages)

# qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1,max_tokens=2048),retriever,qa_prompt=prompt)
qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.1, max_tokens=2048),
    retriever, prompt)

chat_history = []
while True:
    question = input('问题:')
    result = qa.invoke({'question':question,'chat_history':chat_history})
    chat_history.append((question,result['answer']))
    print(result['answer'])

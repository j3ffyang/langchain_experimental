
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
embedding = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
client = QdrantClient("127.0.0.1", port="6333")
qdrant = Qdrant(client, collection_name="chinesemed2", embeddings=embedding)
retriever = qdrant.as_retriever()


# from langchain.memory import VectorStoreRetrieverMemory
# memory = VectorStoreRetrieverMemory(retriever=retriever)


## Define prompt and prompttemplate
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
template = """You are an assistant for question-answering tasks. Use the
following pieces of retrieved context to answer the question. If you don't know
the answer, just say that you don't know. Use three sentences maximum and keep
the answer concise in Chinese.
Question: {question}
Context: {context}
Answer:
"""
# prompt = ChatPromptTemplate.from_template(template)
prompt = PromptTemplate.from_template(template)


# from langchain_community.llms import GPT4All
# llm = GPT4All(
#     model="/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf",
#     device='gpu',
#     n_threads=8)

# from langchain_community.llms import Ollama
# # llm = Ollama(model="gemma:2b")
# llm = Ollama(model="mistral")

# from langchain_community.llms import HuggingFaceEndpoint
# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# llm = HuggingFaceEndpoint(repo_id=repo_id, max_new_tokens=1024, temperature=0.5)

# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# llm = HuggingFacePipeline.from_model_id(
#     model_id="THUDM/chatglm3-6b",
#     task="text-generation",
#     pipeline_kwargs={"max_new_tokens": 100}
# )

# Load model directly
# from transformers import AutoModel
# llm = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(openai_api_base="http://localhost:8000/v1",openai_api_key = "none")


## Create an RAG chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    user_input = input("Enter your question: ")
    if user_input == "exit":
        break
    else:
        print(rag_chain.invoke(user_input))


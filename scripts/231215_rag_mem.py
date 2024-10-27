# https://python.langchain.com/docs/use_cases/question_answering/code_understanding#open-source-llms

# from langchain_community.document_loaders import TextLoader, Docx2txtLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
client = QdrantClient("127.0.0.1", port="6333")
qdrant = Qdrant(client, collection_name="chinesemed", embeddings=embedding)
query = "为什么他们不放过我?"   # error if there is a comma at the end
sim_search = qdrant.similarity_search_with_relevance_scores(query, k=4)
# print(sim_search)
retriever = qdrant.as_retriever()


# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model_name="gpt-4")

# from langchain_community.llms import HuggingFaceEndpoint
# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# llm = HuggingFaceEndpoint(repo_id=repo_id, max_new_tokens=1024, temperature=0.5)

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# model = "google-bert/bert-base-chinese"
model = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model)
# model = AutoModelForCausalLM.from_pretrained(model_id)
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100
)
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0.5})


from langchain.memory import ConversationSummaryMemory
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=False)


from langchain.chains import (
    ConversationalRetrievalChain, LLMChain, StuffDocumentsChain)
chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
print(chain.invoke(query))

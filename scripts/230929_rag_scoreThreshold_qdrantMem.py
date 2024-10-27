# https://stackoverflow.com/questions/77178966/llm-with-vector-database-prompt-to-list-all-stored-documents/77179886#77179886

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.callbacks import StdOutCallbackHandler
from langchain_community.vectorstores import Qdrant
# from langchain_community.vectorstores import FAISS
from qdrant_client import QdrantClient


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # model_name="sentence-transformers/all-mpnet-base-v2"
    )


vectorstore = Qdrant.from_texts(
    ["harry potter's owl is in the castle."],
    embedding = embedding,
    location=":memory:"
)

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 

      {context}

      QUESTION: {question}
      ANSWER:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# llm = HuggingFaceHub(
#     repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
#     # cache_folder="/home/jeff/.cache/torch/sentence_transformers",
#     model_kwargs={'temperature': 0.5, 'max_length': 500}
#     )
llm = HuggingFaceHub(repo_id="EleutherAI/gpt-neo-2.7B")

handler = StdOutCallbackHandler()

qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    retriever=vectorstore.as_retriever(
        # search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.01, "k": 8},
        # search_kwargs={"k": 8}
    ),
    callbacks=[handler],
    # return_source_documents=True
)

print(qa_with_sources_chain)

query = input("Please input your query: ")
response = qa_with_sources_chain.invoke(query)
print(response)


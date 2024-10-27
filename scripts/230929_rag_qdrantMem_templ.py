# https://stackoverflow.com/questions/77178966/llm-with-vector-database-prompt-to-list-all-stored-documents/77179886#77179886

from langchain.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # model_name="sentence-transformers/all-mpnet-base-v2"
    )

# from langchain.vectorstores import Qdrant
# from qdrant_client import QdrantClient
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_texts(
    ["harry potter's owl is in the castle."],
    embedding = embedding,
    # location=":memory:"
)

from langchain.prompts import PromptTemplate
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 

      {context}

      QUESTION: {question}
      ANSWER:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

from langchain.llms import HuggingFaceHub
llm = HuggingFaceHub(
    repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    model_kwargs={'temperature': 0.9, 'max_length': 500}
    )

from langchain.callbacks import StdOutCallbackHandler
handler = StdOutCallbackHandler()

from langchain.chains import RetrievalQA
qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    retriever=vectorstore.as_retriever(
        # search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.01, "k": 8},
    ),
    callbacks=[handler],
    # return_source_documents=True
)

print(qa_with_sources_chain)

query = "where is harry potter's owl?"
response = qa_with_sources_chain.run(query)
print(response)


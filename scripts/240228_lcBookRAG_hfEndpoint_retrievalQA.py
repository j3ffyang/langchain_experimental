from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts(
    ["harry potter's owl is in the castle."],
    embedding = embedding,
)

from langchain.prompts import PromptTemplate
TEMPLATE = """Use the following pieces of context to answer the question at the
end. If you don't know the answer, just say that you don't know, don't try to
make up an answer. Use one single sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Answer:"""
PROMPT = PromptTemplate.from_template(TEMPLATE)

from langchain_community.llms import HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_new_tokens=250, temperature=0.5)


from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_core.vectorstores import VectorStoreRetriever
# retrievalQA = RetrievalQA.from_llm(
#     llm=llm,
#     retriever=retriever,
#     )
retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    retriever = VectorStoreRetriever(vectorstore=vectorstore),
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    callbacks=[StdOutCallbackHandler()],
)

import pprint
pp = pprint.PrettyPrinter(indent=0)
response = retrievalQA.invoke("where is harry potter's owl?")
pp.pprint(response)

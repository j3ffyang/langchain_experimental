
from pprint import pprint

import os
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

from langchain_huggingface import HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id = repo_id,
    temperature = 0.5,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
    max_new_tokens = 250,
)

from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path="./241115_reranker_sample.csv",
                  metadata_columns=["ticket_number", "date", "caller",
                                   "responder", "timestamp"])
data = loader.load()
pprint(data[0])


from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(data, embedding)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

query = "if i forgot the password, how to resolve the problem?"

from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor = compressor, base_retriever = retriever
)
compressed_docs = compression_retriever.invoke(query)
pprint(compressed_docs)


from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm, 
    retriever=compression_retriever,
)
pprint(chain.invoke(query))

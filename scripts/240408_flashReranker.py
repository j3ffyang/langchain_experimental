# https://python.langchain.com/docs/integrations/retrievers/flashrank-reranker/

from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path="./240408_cohereReranker_sample.csv")
data = loader.load()
# print(data)


from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


from langchain_community.vectorstores import FAISS
retriever = FAISS.from_documents(data, embedding).as_retriever(
    search_kwargs={"k": 4})

query = "what about TV"
docs = retriever.get_relevant_documents(query)
# pretty_print_docs(docs)
pprint(docs)

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever)
compressed_docs = compression_retriever.get_relevant_documents(query)
pprint(compressed_docs)


from langchain_community.llms import HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_new_tokens=1024, temperature=0.5)

from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever)

from pprint import pprint
print(chain.invoke(query))

# https://python.langchain.com/docs/integrations/retrievers/flashrank-reranker/

import warnings
warnings.filterwarnings('ignore')

from pprint import pprint
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
from warnings import filterwarnings
# filterwarnings("ignore", category=FutureWarning)
filterwarnings("ignore", category=UserWarning)


from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path="./240409_contextCompression.csv",
                   metadata_columns=["ticket_number", "date", "caller", "responder", "timestamp"])
data = loader.load()
# pprint(data[0])
# pprint(data.Document)


from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


from langchain_community.vectorstores import FAISS
retriever = FAISS.from_documents(data, embedding).as_retriever(
    search_kwargs={"k": 4})

# query = "if i forget the password, how to resolve the problem?"
query = "I'm having trouble logging into my account. I keep getting an error message saying my password is incorrect, even though I know I'm entering it correctly"
# docs = retriever.get_relevant_documents(query)
# pprint(docs)


# from langchain_community.llms import HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_new_tokens=1024, temperature=0.5)
# from langchain_community.llms import GPT4All
# llm = GPT4All(
#     # model="./models/gpt4all/mistral-7b-openorca.Q4_0.gguf",
#     model="/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf",
#     device='gpu',
#     n_threads=8)


from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)
# compressed_docs = compression_retriever.get_relevant_documents(query)
# pprint(compressed_docs)


# from langchain.chains import RetrievalQA
# chain = RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever)
from langchain.chains import RetrievalQAWithSourcesChain
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm, retriever=compression_retriever
)
pprint(chain.invoke(query))


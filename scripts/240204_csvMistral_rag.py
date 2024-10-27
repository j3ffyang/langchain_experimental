# https://stackoverflow.com/questions/77935507/how-to-create-a-chromadb-after-vectorstoreindexcreator-for-your-csv

import shutil
import os
shutil.copy('/home/jeff/Downloads/scratch/instguid.git/lc_tutorial/scripts/240204_csvSample.csv', '/tmp')

from langchain.document_loaders.csv_loader import  CSVLoader
loader = CSVLoader(file_path="/tmp/240204_csvSample.csv",
#                    csv_args={
#                        "delimiter": ",",
#                        "quotechar": '"',
#                        "fieldnames": ["Index", "Organization Id", "Name",
#                                       "Website", "Country", "Description",
#                                       "Founded", "Industry", "Number of employees"],
#                    },
                  )

data = loader.load()
print(data)

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


from langchain_community.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([loader])


from langchain_community.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# Instantiate the model. Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(
    model="/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf",
    device='gpu',
    n_threads=8)


question = "Countries"

from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.vectorstore.as_retriever(),
    # input_key="question"
)

print(chain.invoke("Tell me about countries"))

# print(type(docsearch.vectorstore))
# print(docsearch.vectorstore)

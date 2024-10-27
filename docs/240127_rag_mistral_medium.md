# Document Summarization with Retriever in LangChain

## Objective

To summarize a document using Retrieval Augmented Generation (**RAG**), you can run both VectorStore Embedding and a Large Language Model (LLM) locally. If available, you can also utilize the GPU, such as the Nvidia 4090 in your case.

## RAG Architecture

```plantuml
file doc
rectangle VectorStore {
    component splitter
    component loader
    database embedding
}

actor user

rectangle llm {
    component RetrievalQA.from_chain_type
    component HuggingFacePipeline
}

doc -> VectorStore
loader --> splitter
splitter --> embedding
embedding -> RetrievalQA.from_chain_type
RetrievalQA.from_chain_type --> HuggingFacePipeline
user --> RetrievalQA.from_chain_type
```

## The code step by step

#### Using `WebBaseLoader` to load the document from a web directly

```py
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")
# from langchain.document_loaders import TextLoader, DirectoryLoader
# loader = DirectoryLoader('/tmp/', glob="./*.pdf")
documents = loader.load()
```

I am fond of Hans Andersen's fairy tales, and this particular story still unfolds around us, visible to all.

I also include the code to load document from PDF as above.

#### Split the document and embed it with `sentence-transformers` model from HuggingFace

```py
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                               chunk_overlap=200)
texts = text_splitter.split_documents(documents)

from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # model_name="bert-base-multilingual-cased")
```

As we are using a locally installed open-source embedding model, we have the flexibility to set a large value for the token's `chunk_size` at no cost.

I prefer using the `paraphrase-multilingual-MiniLM-L12-v2 model`, which is 477MB on disk. It is small yet powerful.

#### Save the embedding into VectorStore

```py
from langchain.vectorstores import Chroma
persist_directory = "/tmp/chromadb"
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings,
                                 persist_directory=persist_directory)
vectordb.persist()
```

The database persists in `/tmp/chromadb`. Remind that it will be erased if the system reboots. Therefore, you can change the path if you want to keep the database.

If can also use other VectorStores, such as FAISS, Qdrant, Pinecone that I'll explain in other tutorials later.

#### Specify `PromptTemplate` and `Prompt`

```py
from langchain.prompts import PromptTemplate
prompt_template = """Write a concise summary of the following: "{context}" CONCISE SUMMARY: """
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
```


#### Load `Mistral` model

```py
from langchain_community.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# Instantiate the model. Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(
    model="/home/linux/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf",
    device='gpu',
    n_threads=8)
```

The `mistral` model is downloaded from `https://gpt4all.io/models/gguf/mistral-7b-openorca.Q4_0.gguf`, at https://gpt4all.io/index.html. 


#### Enable debug and verbose mode (optional)
```py
from langchain.globals import set_verbose, set_debug
set_debug(True)
set_verbose(True)
```

Enabling debug and verbose mode can print the detailed chain step by step. It's essential to debug when having issue.

#### RAG with `RetrievalQA` function

```py
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(), chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True)

from langchain.chains import RetrievalQA
response = qa_chain("please summarize this book")
# response = qa_chain("what happened to the emperor?")
```

#### Print the results beautifully (optional)
```py
import pprint
pp = pprint.PrettyPrinter(indent=0)
# pp.pprint(response)
pp.pprint(response['result'])

```

## The complete code

```py
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")
# from langchain.document_loaders import TextLoader, DirectoryLoader
# loader = DirectoryLoader('/tmp/', glob="./*.pdf")
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,
                                               chunk_overlap=200)
texts = text_splitter.split_documents(documents)


from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    # model_name="bert-base-multilingual-cased")


from langchain.vectorstores import Chroma
persist_directory = "/tmp/chromadb"
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings,
                                 persist_directory=persist_directory)
vectordb.persist()


from langchain.prompts import PromptTemplate
prompt_template = """Write a concise summary of the following: "{context}" CONCISE SUMMARY: """
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


from langchain_community.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# Instantiate the model. Callbacks support token-wise streaming
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(
    model="/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf",
    device='gpu',
    n_threads=8)


from langchain.globals import set_verbose, set_debug
set_debug(True)
set_verbose(True)


from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(), chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True)

from langchain.chains import RetrievalQA
response = qa_chain("please summarize this book")
# response = qa_chain("what happened to the emperor?")

import pprint
pp = pprint.PrettyPrinter(indent=0)
# pp.pprint(response)
pp.pprint(response['result'])
```
# https://stackoverflow.com/questions/78813152/ask-multiple-queries-in-parallel-with-langchain-python
# https://github.com/langchain-ai/langchain/discussions/19095


from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")
loader = DirectoryLoader('/tmp/', glob="./*.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(data)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# vectordb = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory='/tmp/chromadb')
vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
retriever = vectordb.as_retriever()


from langchain.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Answer questions using the information in this document and be precise.
Provide only one number if asked for it.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)


from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms import Ollama
llm = Ollama(model="mistral-nemo", base_url="http://127.0.0.1:11434")

def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)


from langchain_core.runnables import RunnableLambda, RunnableParallel

# Define your task as RunnableLambda or any Runnable
task = RunnableLambda(lambda x: x + 1)


parallel_tasks = RunnableParallel({
    'task1': task,
    'task2': task,
})


list_of_dicts = [{'input': 1}, {'input': 2}, {'input': 3}]

# Process each dictionary in parallel
results = []
for input_dict in list_of_dicts:
    result = parallel_tasks.invoke(input_dict['input'])
    results.append(result)

print(results)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke(result)

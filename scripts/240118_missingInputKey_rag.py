# https://python.langchain.com/docs/expression_language/cookbook/retrieval

from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts(
    ["Harry Potter's owl is in the castle"], embedding=embeddings)
retriever = vectorstore.as_retriever()


# from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
template = """Answer the question based on on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


from langchain_community.llms import GPT4All
model = GPT4All(
    model="/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf",
    device='gpu',
    n_threads=8)


from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


from langchain.globals import set_verbose, set_debug
set_debug(True)
set_verbose(True)

result = chain.invoke("Where is the hedwig?")
print(result)

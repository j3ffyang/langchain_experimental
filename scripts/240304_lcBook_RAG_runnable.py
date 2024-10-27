# https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")
document = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(document)

from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()


from langchain.prompts import ChatPromptTemplate
template = """You are an assistant for question-answering tasks. Use the
following pieces of retrieved context to answer the question. If you don't know
the answer, just say that you don't know. Use three sentences maximum and keep
the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)


# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# model_id = "gpt2"
# model_id = "distilbert/distilgpt"
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# hf = HuggingFacePipeline(pipeline=pipe)


from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    # model_id="google/flan-t5-small",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100}
)


# from langchain_community.llms import HuggingFaceEndpoint
# llm = HuggingFaceEndpoint(
#     repo_id="google/flan-t5-xxl", max_length=500, max_new_tokens=250, temperature=0.5
# )


from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# query = "Please summarize the world history"
query = "Please tell me the history about world war tow"
print(rag_chain.invoke(query))

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Text_file")
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)


from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
# url = "127.0.0.1:6333"
# qdrant = Qdrant.from_documents(
#     docs,
#     embedding,
#     url=url,
#     collection_name="wikipedia",
# )
qdrant = Qdrant.from_documents(
    docs,
    embedding,
    location=":memory:",
    collection_name="wikipedia",
)
retriever = qdrant.as_retriever()


from langchain.prompts import PromptTemplate
prompt_template = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>

 """
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)


from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# model_id = "google/flan-t5-small"
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.5,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=256
)
llm = HuggingFacePipeline(pipeline=pipe)

from langchain_core.output_parsers import StrOutputParser
llm_chain = prompt | llm | StrOutputParser()


question = "what is the flat file?"   # error if there is a comma at the end
from langchain_core.runnables import RunnablePassthrough
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain

# print(llm_chain.invoke({"context": "", "question": question}))
print(rag_chain.invoke(question))

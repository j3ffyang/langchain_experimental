# https://python.langchain.com/docs/expression_language/cookbook/retrieval

# from langchain.chat_models import ChatOpenAI
# model = ChatOpenAI()

from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts(
    ["harry potter's owl is in the castle", "harry potter's owl is in the castle"],
    embedding = embedding,
    )
retriever = vectorstore.as_retriever()

from langchain.prompts import ChatPromptTemplate
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# from langchain_community.llms.huggingface_hub import HuggingFaceHub
# model = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")

from langchain_community.llms import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-xxl", max_length=500, max_new_tokens=250, temperature=0.5
)


from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke("where is harry potter's hedwig?"))

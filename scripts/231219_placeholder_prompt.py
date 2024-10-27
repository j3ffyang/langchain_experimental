# https://stackoverflow.com/questions/77681529/how-to-change-the-placeholder-in-prompt

from langchain.prompts import ChatPromptTemplate
template = """Answer the question based on on the following context:
    {context}

    Question:
    """


from langchain.prompts import MessagesPlaceholder
# from langchain.prompts import HumanMessagePromptTemplate
# from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="system"),
        ("user", "{question}")
    ]
)


from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts(
    ["harry potter's owl is in the castle"],
    embedding = embedding,
    )
retriever = vectorstore.as_retriever()


from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from langchain.prompts import PromptTemplate
chain = (
    RunnablePassthrough.assign(content=lambda x:
                                 retriever.invoke(x["question"]))
    | RunnablePassthrough.assign(system=lambda x:
#                                [SystemMessage(content=template)])
                               [SystemMessage(content=PromptTemplate.from_template(template).format(context=x["content"][0].page_content))])
    | prompt
)

result = chain.invoke({"question": "where is harry potter's hedwig?"})
print(result)

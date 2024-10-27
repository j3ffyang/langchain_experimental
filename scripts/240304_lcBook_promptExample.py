from langchain.prompts import ChatPromptTemplate
template = """You are an assistant for question-anwering tasks. Use the
following pieces of retrieved context to answer the question. If you don't know
the answer, just say that you don't know. Use three sentences maximum and keep
the answer concise.
Question: {question}
Context: {context}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
print(prompt)

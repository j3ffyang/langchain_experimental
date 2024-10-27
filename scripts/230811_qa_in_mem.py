# https://python.langchain.com/docs/use_cases/question_answering/how_to/question_answering
# QA over in-memory documents


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator


orig_file = "/home/jeff/Downloads/scratch/instguid.git/hlmGPT/jeff_tutorials/data/books/epub_to_txt/红楼十二层：周汝昌妙解红楼.txt"

with open(orig_file, "r") as f:
    text = f.read()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_text(text)

embeddings = OpenAIEmbeddings()

docsearch = Chroma.from_texts(docs, embeddings, metadatas=[{"source": str(i)} for i in range(len(docs))]).as_retriever()

# query = "周汝昌如何解讀紅樓夢？"
query = "林黛玉和賈寶玉什麼關係？"
docs = docsearch.get_relevant_documents(query)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(temperature=0.5), chain_type="refine")
print(chain.run(input_documents=docs, question=query))

# prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
# 
# {context}
# 
# Question: {question}
# Answer in Chinese:"""
# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )
# chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt=PROMPT)
# print(chain({"input_documents": docs, "question": query}, return_only_outputs=True))


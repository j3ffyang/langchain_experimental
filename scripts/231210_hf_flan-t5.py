## https://stackoverflow.com/questions/77633352/chatbox-is-returning-one-word-answer-instead-of-full-sentence

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

# embeddings = HuggingFaceEmbeddings(model_name='bert-base-uncased')
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# docsearch = FAISS.from_documents(texts, embeddings)
docsearch = FAISS.from_texts(
    ["harry potter's owl is in the castle. The book is about 'To Kill A Mocking Swan'. There is another monkey"], embeddings)


from langchain_community.llms import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-xxl", max_length=500, max_new_tokens=250, temperature=0.5
)

prompt_template = """
Compare the book given in question with others in the retriever based on genre and description.
Return a complete sentence with the full title of the book and describe the similarities between the books.

question: {question}
context: {context}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
retriever=docsearch.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs = {"prompt": prompt})
print(qa.invoke({"query": "Which book except 'To Kill A Mocking Bird' is similar to it?"}))

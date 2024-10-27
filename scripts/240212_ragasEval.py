# https://docs.ragas.io/en/v0.0.18/howtos/integrations/langchain.html
# https://colab.research.google.com/drive/1C1Epju1lVkXTQi2jBq1njrOrmkfg0eQS?usp=sharing#scrollTo=-TsjUWjbUfbW

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")

documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

documents = text_splitter.split_documents(documents)
print(documents)
print(len(documents))


from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

generator = TestsetGenerator.with_openai()
# generator = TestsetGenerator.generate_with_langchain_docs()
testset = generator.generate_with_langchain_docs(documents,
                                                 test_size=10,
                                                 distributions={simple: 0.5,
                                                                reasoning: 0.25,
                                                                multi_context: 0.25})
print(testset.test_data[0])

test_df = testset.to_pandas()
print(test_df)

# from langchain.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# 
# 
# from langchain_community.vectorstores import FAISS
# vectorstore = FAISS.from_documents(documents, embeddings)
# retriever = vectorstore.as_retriever()
# 
# retriever_documents = retriever.invoke("What did the emperor say?")
# for doc in retriever_documents:
#     print(doc)
# 
# 
# from langchain import hub
# retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
# print(retrieval_qa_prompt.messages[0].prompt.template)
# 
# 
# from langchain.prompts import ChatPromptTemplate
# template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':
# 
# Context:
# {context}
# 
# Question:
# {question}
# """
# 
# prompt = ChatPromptTemplate.from_template(template)
# print(prompt)
# 
# 
# from operator import itemgetter
# from langchain_community.llms import GPT4All
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# callbacks = [StreamingStdOutCallbackHandler()]
# primary_qa_llm = GPT4All(
#     model="/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf",
#     device='gpu',
#     n_threads=8)
# 
# 
# from langchain_core.runnables import RunnablePassthrough
# retrieval_augmented_qa_chain = (
#     # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
#     # "question" : populated by getting the value of the "question" key
#     # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
#     {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
#     # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
#     #              by getting the value of the "context" key from the previous step
#     | RunnablePassthrough.assign(context=itemgetter("context"))
#     # "response" : the "context" and "question" values are used to format our prompt object and then piped
#     #              into the LLM and stored in a key called "response"
#     # "context"  : populated by getting the value of the "context" key from the previous step
#     | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
# )
# 
# question = "What is the main story in this article?"
# result = retrieval_augmented_qa_chain.invoke({"question": question})
# # print((result["response"].content)
# print(result["response"])
# 

# see https://www.youtube.com/watch?v=TLf90ipMzfE

from PyPDF2 import PdfReader
reader = PdfReader('/home/jeff/Downloads/231214_linux_refdoc.pdf')

# read date form the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
     text = page.extract_text()
     if text:
         raw_text += text
# print(raw_text[:100])

from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
)
chunks = splitter.split_text(raw_text)


from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


from langchain_community.vectorstores import FAISS
vectorstore = FAISS.from_texts(chunks, embedding)


from langchain_community.llms import HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_new_tokens=1024, temperature=0.5)


from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff")
question = "who are the authors of the article?"
input  = vectorstore.similarity_search(query)
print(chain.run(input_documents=input, question=question))

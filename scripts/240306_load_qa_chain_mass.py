# https://stackoverflow.com/questions/78284714/how-to-fix-invalidrequesterror-the-model-text-davinci-003-has-been-deprecated

from PyPDF2 import PdfReader
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

pdfs_folder = '/tmp'
pdf_files = [file for file in os.listdir(pdfs_folder) if file.endswith('.pdf')]
raw_text = ''
# Iterate through each PDF file
for pdf_file in pdf_files:
    # Construct the path to the PDF file
    pdf_path = os.path.join(pdfs_folder, pdf_file)
    # Read the PDF file
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        # Extract raw text from pages
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 500,
    chunk_overlap  = 200,
    length_function = len,
)

chunks = text_splitter.split_text(raw_text)
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(chunks, embeddings)

from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(OpenAI(), chain_type="stuff")
# query = "What are the core values of JM Finance?"
query = "what about the emperor"
docs = docsearch.similarity_search(query)
print(chain.run(input_documents=docs, question=query))

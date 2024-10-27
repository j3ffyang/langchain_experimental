# https://gptcache.readthedocs.io/en/latest/bootcamp/langchain/qa_generation.html

from langchain.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

ORIG_FILE = "/home/jeff/Downloads/scratch/instguid.git/hlmGPT/jeff_tutorials/data/books/epub_to_txt/红楼十二层：周汝昌妙解红楼.txt"

loader = TextLoader(ORIG_FILE)

doc = loader.load()[0]


chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0), text_splitter=text_splitter)

qa = chain.run(doc.page_content)
print(qa)

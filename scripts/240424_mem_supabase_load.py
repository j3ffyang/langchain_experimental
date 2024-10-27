# https://supabase.com/docs/guides/self-hosting/docker
# https://www.linode.com/docs/guides/installing-supabase/
# https://python.langchain.com/docs/integrations/vectorstores/supabase/


from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")
documents = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)


from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


import os
import getpass
os.environ["SUPABASE_URL"] = getpass.getpass("Supabase URL: ")
os.environ["SUPABASE_SERVICE_KEY"] = getpass.getpass("Supabase Svc Key: ")


from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

vectorstore = SupabaseVectorStore.from_documents(
    chunks,
    embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    chunk_size=500,
)

# query = "What does the emperor think about his cloth?"
# print(vectorstore.similarity_search(query, k=3))

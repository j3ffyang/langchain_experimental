from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import argparse, pprint


def args():
    # argparse
    parser = argparse.ArgumentParser(description='Initialize Chroma database')
    parser.add_argument('--book', type=str, required=False, help='book for init')
    parser.add_argument("--flag", choices=["init", "get", "add"], default="get")
    args = parser.parse_args()
    return args


def split(book):
    # loader = TextLoader(book_init)
    loader = TextLoader(book)
    documents = loader.load()

    # split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


# Init Chroma. Notice: this will wipe out the database
def init(docs, embedding_function):
    db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")


# Get Chroma = query
def get(embedding_function):
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    print(db.get().keys())
    print(len(db.get()["ids"]))
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(db.get())


# Add document into Chroma
def add(docs, embedding_function):
    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    db.add_documents(docs)


def main():
    flag = args().flag
    embedding_function = OpenAIEmbeddings()

    if flag == "init":
       docs = split(args().book)
       init(docs, embedding_function)

    elif flag == "get":
        get(embedding_function)

    elif flag == "add":
        docs = split(args().book)
        add(docs, embedding_function)


if __name__ == "__main__":
    main()


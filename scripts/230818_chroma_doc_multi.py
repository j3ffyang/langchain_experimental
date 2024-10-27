from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
import argparse, pprint


def args():
    # argparse
    parser = argparse.ArgumentParser(description='Initialize Chroma database')
    parser.add_argument('--dir', type=str, required=False, help='Load multiple books')
    parser.add_argument("--flag", choices=["init", "get", "add"], default="get")
    args = parser.parse_args()
    return args


def split(dir):
    loader = DirectoryLoader(dir, glob="**/*.md", show_progress=True, use_multithreading=True)
    documents = loader.load()

    # split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


def add(docs, embedding_function):
    chroma = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    chroma.add_documents(docs)


def get(embedding_function):
    chroma = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
    chroma.get()


def main():
    flag = args().flag
    embedding_function = OpenAIEmbeddings()

    if flag == "get":
        get(embedding_function)

    elif flag == "add":
        docs = split(args().dir)
        add(docs, embedding_function)


if __name__ == "__main__":
    main()
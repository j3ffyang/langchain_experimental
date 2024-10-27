# https://qdrant.tech/documentation/integrations/langchain/

import argparse
from pprint import pprint
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from qdrant_client import QdrantClient


def argparser():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--flag", type=str, required=False)
    args = parser.parse_args()
    return args


def splitter(book):
    """Define load document and split into chunks"""
    # BOOK = "/home/jeff/Downloads/scratch/instguid.git/hlmGPT/data/books/红楼艺术_周汝昌.txt"
    loader = TextLoader(book)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=500,
    #     chunk_overlap=50,
    #     separators=["\n\n", "\n", " ", ""],
    #     )
    docs = text_splitter.split_documents(documents)
    return docs


def init(docs, embeddings):
    """Initialize Qdrant database"""
    # client = QdrantClient("127.0.0.1", port="6333")
    url = "http://127.0.0.1:6333"
    qdrant = Qdrant.from_documents(
        docs,
        embeddings,
        url=url,
        # prefer_grpc=True, # requires GRPC service running over 6334
        collection_name="hlm",
    )
    return qdrant


def get():
    """Just query collection_name"""
    client = QdrantClient("127.0.0.1", port="6333")
    collection_info = client.get_collection(collection_name="hlm")
    print(collection_info)


def add(docs, embeddings):
    """Add documents to collection_name"""
    client = QdrantClient("127.0.0.1", port="6333")
    qdrant = Qdrant(client, collection_name="hlm", embeddings=embeddings)
    qdrant.add_documents(docs)


def sim_search(embeddings, query):
    """Query collection_name"""
    # query = "贾宝玉和晴雯的关系？"
    client = QdrantClient("127.0.0.1", port="6333")
    qdrant = Qdrant(client, collection_name="hlm", embeddings=embeddings)
    # found_docs = qdrant.similarity_search(query, k=8)   # top_k setting
    found_docs = qdrant.similarity_search_with_relevance_scores(query, k=8)   # top_k setting
    return found_docs
    # print(found_docs[0].page_content)


def main():
    """Main function"""
    args = argparser()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
        # model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    if args.flag == "init":
        book = input("Please enter the book path and name: ")
        docs = splitter(book)
        init(docs, embeddings)
    elif args.flag == "add":
        book = input("Please enter the book path and name: ")
        docs = splitter(book)
        add(docs, embeddings)
    elif args.flag == "get":
        get()
    elif args.flag == "query":
        query = input("Please enter query: ")
        search = sim_search(embeddings, query)
        # search = found_docs(args.query)
        pprint(search)
    else:
        print("No flag specified")


if __name__ == "__main__":
    main()

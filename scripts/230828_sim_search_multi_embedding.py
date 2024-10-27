"""Use diff embeddings to vectorize and similarity_search from Qdrant"""

import argparse
from pprint import pprint
import pandas as pd
# from langchain.document_loaders import UnstructuredExcelLoader, TextLoader
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient


def argsparser():
    """parser.parse setting"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--flag", type=str, required=False, help="This is help")
    args = parser.parse_args()
    return args


def text_load(excel_file):
    """Load excel into documents for text_loader"""
    # excel_file = "/home/jeff/Downloads/scratch/instguid.git/hlmGPT/tutorials/data/ticketing/ticket_event_orig2.xls"
    df = pd.read_excel(excel_file, usecols='D')   # Read column D from excel
    documents = []
    for index, row in df.iterrows():
    # for index, row in df.loc[df['ColumnName'] != 'content'].iterrows():
        # print(row.values)
        documents.append(str(row.values))
    # print(documents)
    # print(type(documents))
    # print(len(documents))
    return documents


def create_hf(documents):
    """Initialize Qdrant database with Hugging Face embedding"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # client = QdrantClient("127.0.0.1", port="6333") # Connect to Qdrant server
    qdrant = Qdrant.from_texts( 
        documents,
        embeddings,
        # path="./qdrant_db",
        host="localhost",
        collection_name="hf_collection",
    )
    # qdrant.from_text(documents)
    return qdrant


def create_openai(documents):
    """Init Qdrant database with OpenAI embedding"""
    embeddings = OpenAIEmbeddings()
    qdrant = Qdrant.from_texts(
        documents,
        embeddings,
        # path="./qdrant_db",
        host="localhost",
        collection_name="openai_collection",
    )
    return qdrant


def get_collection():
    """List the existing collections in Qdrant database"""
    client = QdrantClient("127.0.0.1", port="6333") # Connect to Qdrant server
    # client = QdrantClient(path="./qdrant_db")
    collections = client.get_collections()
    pprint(collections, depth=1)
    print(collections)

    # for collection in collections:
    #     print(collection.name, collection.status, collection.points_count)
    # return collections


def search_hf(query):
    """Query collection_name with HuggingFace"""
    # query = "养猪通密碼忘記怎麼辦？"

    # client = QdrantClient(path="./qdrant_db")
    embedding_hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # client = QdrantClient(path="./qdrant_db")
    client = QdrantClient("127.0.0.1", port="6333") # Connect to Qdrant server
    qdrant_hf = Qdrant(client, collection_name="hf_collection", embeddings=embedding_hf)
    # found_docs = qdrant.similarity_search(query)
    found_hf = qdrant_hf.similarity_search_with_relevance_scores(query)
    # print(found_docs[0].id)
    # print(found_docs[0].score)
    # print(found_docs[0].page_content)
    # print(found_docs)
    return found_hf


def search_oa(query):
    """Query collection_name with OpenAI"""
    # client = QdrantClient(path="./qdrant_db")
    client = QdrantClient("127.0.0.1", port="6333") # Connect to Qdrant server
    embedding_oa = OpenAIEmbeddings()
    qdrant_oa = Qdrant(client, collection_name="openai_collection", embeddings=embedding_oa)
    found_oa = qdrant_oa.similarity_search_with_relevance_scores(query)
    return found_oa


def main():
    """Main func"""
    args = argsparser()

    if args.flag == "create_hf":
        excel_file = input("Please enter Excel file with the exact path: ")
        documents = text_load(excel_file)
        create_hf(documents)

    elif args.flag == "create_oa":
        excel_file = input("Please enter Excel file with the exact path: ")
        documents = text_load(excel_file)
        create_openai(documents)

    elif args.flag == "query_hf":
        query = input("Please enter query: ")
        print(search_hf(query))

    elif args.flag == "query_oa":
        query = input("Please enter query: ")
        print(search_oa(query))

    elif args.flag == "get":
        get_collection()

    # add()
    # get()

    else:
        print("No flag specified!")


if __name__ == '__main__':
    main()

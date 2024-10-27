# https://stackoverflow.com/questions/77178966/llm-with-vector-database-prompt-to-list-all-stored-documents/77179886#77179886

import argparse
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub, OpenAI
from langchain.callbacks import StdOutCallbackHandler
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient


def arg_parse():
    """Parse command line argument"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--flag", type=str, required=False)
    args = parser.parse_args()
    return args


def text_load(excel_file):
    """Load text from XLS file"""
# excel_file = "/home/jeff/Downloads/scratch/instguid.git/hlmGPT/tutorials/data/recipe/10_intl_recipe.xls"
    # df = pd.read_excel(excel_file, skiprows=[0])
    df = pd.read_excel(excel_file)
    documents = []
    for index, row in df.iterrows():
        documents.append(str(row.values))
    print(documents)
    print(type(documents))
    print(len(documents))
    return documents


def init(documents, embeddings):
    """Init Qdrant database"""
    url = "http://127.0.0.1:6333"
    qdrant = Qdrant.from_texts(
        documents,
        embeddings,
        url=url,
        collection_name="recipes",
    )
    return qdrant


def sim_search(embeddings, query):
    """Define vector_store"""
    client = QdrantClient("127.0.0.1", port="6333")
    vector_store = Qdrant(client, collection_name="recipes", embeddings=embeddings)
    found_docs = vector_store.similarity_search_with_relevance_scores(query, k=4)
    return found_docs


def main():
    """Main func"""

    args = arg_parse()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        # model_name="sentence-transformers/all-mpnet-base-v2"
    )

    if args.flag == "text_load":
        excel_file = input("Please give me your recipe list: ")
        text_load(excel_file)

    if args.flag == "init":
        excel_file = input("Please give me your recipe list: ")
        documents = text_load(excel_file)
        init(documents, embeddings)

    if args.flag == "query":
        client = QdrantClient("127.0.0.1", port="6333")
        vector_store = Qdrant(client, collection_name="recipes", embeddings=embeddings)

        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. 

      {context}

      QUESTION: {question}
      ANSWER: 
        """

        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # llm = OpenAI()
        llm = HuggingFaceHub(
            repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
            # repo_id="openai-gpt", # doesn't work
            # cache_folder="/home/jeff/.cache/torch/sentence_transformers",
            model_kwargs={'temperature': 0.3, 'max_length': 500}
            )

        handler = StdOutCallbackHandler()

        qa_with_sources_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            chain_type_kwargs={"prompt": PROMPT},
            retriever=vector_store.as_retriever(
                # search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.01, "k": 8},
                # search_kwargs={"k": 8}
            ),
            callbacks=[handler],
            # return_source_documents=True
        )

        # pprint(qa_with_sources_chain)

        query = input("Please input your query: ")
        response = qa_with_sources_chain.run(query)
        print(response)


if __name__ == '__main__':
    main()

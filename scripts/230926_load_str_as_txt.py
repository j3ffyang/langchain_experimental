# https://stackoverflow.com/questions/77045559/langchain-load-with-string/77201180#77201180

from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document

def get_text_chunks_langchain(text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs


def main():
    text = "If you want to output the query's result as a string, keep in mind that LangChain retrievers give a Document object as output. Therefore, your function should look like this:"
    docs = get_text_chunks_langchain(text)
    print(docs)


if __name__ == '__main__':
    main()

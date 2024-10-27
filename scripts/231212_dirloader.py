# https://stackoverflow.com/questions/77645110/iterate-on-langchain-document-items/77645658#77645658

# from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader, DirectoryLoader
loader = DirectoryLoader('/tmp/', glob="**/*.md", loader_cls=TextLoader,
                         show_progress=True, use_multithreading=True)
documents = loader.load()
print(len(documents))

# for doc in documents:
#     texts = text_splitter.split_documents(doc)
#     chain = load_summarize_chain(llm, chain_type="map_reduce",
#                                  map_prompt=prompt, combine_prompt=prompt)

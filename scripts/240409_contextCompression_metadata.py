# https://python.langchain.com/docs/integrations/retrievers/flashrank-reranker/

from pprint import pprint

from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path="./240409_contextCompression.csv",
                   metadata_columns=["ticket_number", "date", "caller",
                                     "responder", "timestamp"])
# loader = CSVLoader(file_path="./240409_contextCompression.csv", source_column="solution")
# loader = CSVLoader(file_path="./240409_contextCompression.csv")
data = loader.load()
# pprint(type(data[0]))
pprint((data[0]))
# pprint(data["Document"])

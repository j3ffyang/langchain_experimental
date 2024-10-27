# https://stackoverflow.com/questions/78022298/modulenotfounderror-no-module-named-langchain

from langchain_community.document_loaders.image import UnstructuredImageLoader
loader = UnstructuredImageLoader("/home/jeff/240216_monalisa_baozi.png",
                                # mode="elements", strategy="fast",
                                 mode="single",
)
# loader = UnstructuredImageLoader("/home/jeff/Downloads/240205_tecumsethMart_holdingStop.jpeg")
data = loader.load()
print(data[0])

# https://stackoverflow.com/questions/77625508/how-to-activate-verbosity-in-langchain/77629872#77629872

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.globals import set_verbose, set_debug

set_debug(True)
set_verbose(True)

prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser
print(chain.invoke({"topic": "ice cream"}))

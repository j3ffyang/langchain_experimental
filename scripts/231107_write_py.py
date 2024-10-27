# https://python.langchain.com/docs/expression_language/cookbook/code_writing

from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
from langchain_experimental.utilities import PythonREPL

template = """Write some python code to solve the user's problem

Return only python code in Markdown format, e.g.:

```python
....
```"""
prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])

model = ChatOpenAI()

def _santitize_output(text: str):
    _, after = text.split("```python")
    return after.split("```")[0]

chain = prompt | model | StrOutputParser() | _santitize_output | PythonREPL().run

print(chain.invoke({"input": "whats 2 plus 2 in python"}))

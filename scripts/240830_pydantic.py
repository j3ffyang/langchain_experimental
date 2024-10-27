# https://stackoverflow.com/questions/78904911/pydantic-is-not-compatible-with-langchain-documents/78929028#78929028

# %%
from typing import List

from langchain_core.documents.base import Document
from pydantic import BaseModel, ConfigDict

from pydantic.v1 import BaseModel # <-- Note v1 namespace

class ResponseBody(BaseModel):
    message: List[Document]
    model_config = ConfigDict(arbitrary_types_allowed=True)

docs = [Document(page_content="This is a document")]
res = ResponseBody(message=docs)
# %%

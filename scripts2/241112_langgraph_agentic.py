# https://python.langchain.com/docs/tutorials/qa_chat_history/#agents

import getpass
import os

def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")

_set_env("HUGGINGFACEHUB_API_TOKEN")

print("HUGGINGFACEHUB_API_TOKEN:",
      os.environ.get("HUGGINGFACEHUB_API_TOKEN"))

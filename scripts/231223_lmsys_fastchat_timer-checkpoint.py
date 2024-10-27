# https://www.markhneedham.com/blog/2023/06/23/hugging-face-run-llm-model-locally-laptop/

from langchain.llms import HuggingFacePipeline, HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# model_id = "lmsys/fastchat-t5-3b-v1.0"
# llm = HuggingFacePipeline.from_model_id(
#     model_id=model_id,
#     task="text2text-generation",
#     model_kwargs={"temperature": 0, "max_length": 1000},
# )

llm = HuggingFaceHub(repo_id = "lmsys/fastchat-t5-3b-v1.0",
                     model_kwargs={"temperature": 0.3, "max_length": 1000})

template = """
You are a friendly chatbot assistant that responds conversationally to users'
questions. Keep the answers short, unless specifically asked by the user to
elaborate on something.

Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

def ask_question(question):
    result = llm_chain(question)
    print(result['question'])
    print("")
    print(result['text'])


import time

class TimeError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def __enter__(self):
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        self._start_time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")

with Timer():
    ask_question("Describe some famous landmarks in London")


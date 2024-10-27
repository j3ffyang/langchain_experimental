import openai
import os
from langchain.llms import OpenAI
from langchain import FewShotPromptTemplate
from langchain import PromptTemplate

openai.api_key = os.getenv("OPENAI_API_KEY")

openai = OpenAI(
    model_name="text-davinci-003",
    openai_api_key = openai.api_key
)

examples = [
    {
        "query": "How are you?",
        "answer": "I can't complain but sometimes I still do."
    }, {
        "query": "What time is it?",
        "answer": "It's time to get a watch."
    }
]

example_template = """
User: {query}
AI: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

prefix = """The following are exerpts from conversations with an AI assistent. The assistant is typically sarcastic and witty, producing creative and funny responses to the users questions. Here are some examples:
"""

suffix = """
User: {query}
AI: """

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator='\n\n'
)

query = "What is the meaning of life?"

print(few_shot_prompt_template.format(query=query))
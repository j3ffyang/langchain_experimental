# https://stackoverflow.com/questions/78584114/best-way-to-stream-using-langchain-llm-with-structured-output

import asyncio
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")

# Define the model with structured output
model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)
structured_llm = model.with_structured_output(Joke)

def stream_joke():
    # Invoke the model and stream the response
    response = structured_llm.stream("Tell me a joke about cats")

    # Initialize an empty Joke object
    joke = Joke(setup="", punchline="")

    # Stream the response
    for part in response:
        if 'setup' in part:
            joke.setup += part['setup']
            print(f"Setup: {joke.setup}")
        if 'punchline' in part:
            joke.punchline += part['punchline']
            print(f"Punchline: {joke.punchline}")

# Run the streaming joke function
print(stream_joke())

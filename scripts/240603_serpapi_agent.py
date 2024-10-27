#  https://stackoverflow.com/questions/77391069/parsing-error-on-langchain-agent-with-gpt4all-llm/78573136#78573136

from langchain.llms import GPT4All
from langchain.agents import load_tools
from dotenv import load_dotenv

load_dotenv()

llm = GPT4All(
    # model="/home/jeff/.cache/gpt4all/orca-mini-3b-gguf2-q4_0.gguf",
    model="/home/jeff/.cache/gpt4all/mistral-7b-openorca.gguf2.Q4_0.gguf",
    verbose=True,
)

from langchain.agents import load_tools
from langchain.agents import initialize_agent

tools = load_tools(["serpapi"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True,
                        handle_parsing_errors=True)
agent.invoke("which club is Cristiano Ronaldo playing right now?")

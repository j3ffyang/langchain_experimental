# https://langchain-ai.github.io/langgraph/tutorials/introduction/#setup

import getpass
import os


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getuser(f"{var}: ")


_set_env("ANTHROPIC_API_KEY")
_set_env("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangGraph Tutorial"


from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

class State(TypedDict):
    # Messages have the type "list". The `add_messages`` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the lists, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)


from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-haiku-20240307")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever 
# the node is used
graph_builder.add_node("chatbot", chatbot)

graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")

graph = graph_builder.compile()

# from asciichartpy import plot
# from IPython.display import Image, display
# try:
#     display(Image(graph.get_graph().draw_ascii()))
# except:
#     # This requires some extra dependencies and is optional
#     pass
# 
# print(plot(display(Image(graph.get_graph().draw_ascii()))))

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# https://stackoverflow.com/questions/78967456/issue-in-running-a-plan-and-execute-based-llm-agent-in-python-code-hangs

# %% 
import getpass
import os
from langchain_openai import ChatOpenAI
from typing import Sequence

from langchain import hub
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from output_parser_v2 import LLMCompilerPlanParser, Task

from langchain_community.tools import DuckDuckGoSearchRun

from math_tools import get_math_tool

#Task fetching unit imports
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Dict, Iterable, List, Union

from langchain_core.runnables import (
    chain as as_runnable,
)
from typing_extensions import TypedDict
import itertools

#Joiner imports
from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# os.environ['OPENAI_API_KEY'] = "<insert key here>"

def _get_pass(var: str):
    if var not in os.environ:
        os.environ[var] = getpass.getpass(f"{var}: ")


def tools():
    calculate = get_math_tool(ChatOpenAI(model="gpt-4-turbo-preview"))
    math = DuckDuckGoSearchRun()
    tool = DuckDuckGoSearchRun(
        name="Web-tool",
        description="look up things in wikipedia",
    )
    tools = [tool, calculate]
    
    return tools
    
# This is for the replacement of prompt from langchain hub - first set of hub.pull() replacement
def joiner():

    # prompt = hub.pull("wfh/llm-compiler")

    system_prompt_1 = """
    Solve a question answering task. Here are some guidelines:
 - In the Assistant Scratchpad, you will be given results of a plan you have executed to answer the user's question.
 - Thought needs to reason about the question based on the Observations in 1-2 sentences.
 - Ignore irrelevant action results.
 - If the required information is present, give a concise but complete and helpful answer to the user's question.
 - If you are unable to give a satisfactory finishing answer, replan to get the required information. Respond in the following format:

Thought: <reason about the task results and whether you have sufficient information to answer the question>
Action: <action to take>
Available actions:
 (1) Finish(the final answer to return to the user): returns the answer and finishes the task.
 (2) Replan(the reasoning and other information that will help you plan again. Can be a line of any length): instructs why we must replan
    """

    system_prompt_2 = """Using the above previous actions, decide whether to replan or finish. If all the required information is present. You may finish. If you have made many attempts to find the information without success, admit so and respond with whatever information you have gathered so the user can work well with you.

{examples}"""

    placeholder_prompt = """{messages}"""

    full_prompt = system_prompt_1 + placeholder_prompt + system_prompt_2
    prompt = PromptTemplate(
        input_variables = ["examples","messages"],
        template = full_prompt
    )
    #Below often gets printed as None
    print("PLANNER FUNC PRINT = {}".format(prompt.pretty_print()))
    return prompt

# 2nd first set of hub.pull() replacement
def planner():

    # prompt = hub.pull("wfh/llm-compiler")

    system_prompt_1 = """
    Given a user query, create a plan to solve it with the utmost parallelizability. Each plan should comprise an action from the following {num_tools} types:
{tool_descriptions}
{num_tools}. join(): Collects and combines results from prior actions.

 - An LLM agent is called upon invoking join() to either finalize the user query or wait until the plans are executed.
 - join should always be the last action in the plan, and will be called in two scenarios:
   (a) if the answer can be determined by gathering the outputs from tasks to generate the final response.
   (b) if the answer cannot be determined in the planning phase before you execute the plans. Guidelines:
 - Each action described above contains input/output types and description.
    - You must strictly adhere to the input and output types for each action.
    - The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.
 - Each action in the plan should strictly be one of the above types. Follow the Python conventions for each action.
 - Each action MUST have a unique ID, which is strictly increasing.
 - Inputs for actions can either be constants or outputs from preceding actions. In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.
 - Always call join as the last action in the plan. Say '<END_OF_PLAN>' after you call join
 - Ensure the plan maximizes parallelizability.
 - Only use the provided action types. If a query cannot be addressed using these, invoke the join action for the next steps.
 - Never introduce new actions other than the ones provided.
    """

    system_prompt_2 = """Remember, ONLY respond with the task list in the correct format! E.g.:
idx. tool(arg_name=args)"""

    placeholder_prompt = """{messages}"""

    full_prompt = system_prompt_1 + placeholder_prompt + system_prompt_2
    prompt = PromptTemplate(
        input_variables = ["tool_descriptions","num_tools","messages"],
        template = full_prompt
    )

    # Same None output for below
    print("PLANNER FUNC PRINT = {}".format(prompt.pretty_print()))
    return prompt


def create_planner(
    llm: BaseChatModel, tools: Sequence[BaseTool], base_prompt: ChatPromptTemplate
):
    tool_descriptions = "\n".join(
        f"{i+1}. {tool.description}\n"
        for i, tool in enumerate(
            tools
        )  # +1 to offset the 0 starting index, we want it count normally from 1.
    )
    planner_prompt = base_prompt.partial(
        replan="",
        num_tools=len(tools)
        + 1,  # Add one because we're adding the join() tool at the end.
        tool_descriptions=tool_descriptions,
    )
    replanner_prompt = base_prompt.partial(
        replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
        "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
        'You MUST use these information to create the next plan under "Current Plan".\n'
        ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
        " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
        " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )

    def should_replan(state: list):
        # Context is passed as a system message
        return isinstance(state[-1], SystemMessage)

    def wrap_messages(state: list):
        return {"messages": state}

    def wrap_and_get_last_index(state: list):
        next_task = 0
        for message in state[::-1]:
            if isinstance(message, FunctionMessage):
                next_task = message.additional_kwargs["idx"] + 1
                break
        state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
        return {"messages": state}

    return (
        RunnableBranch(
            (should_replan, wrap_and_get_last_index | replanner_prompt),
            wrap_messages | planner_prompt,
        )
        | llm
        | LLMCompilerPlanParser(tools=tools)
    )

# Task Fetching Unit

def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    # Get all previous tool responses
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    return results


class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterable[Task]


def _execute_task(task, observations, config):
    tool_to_use = task["tool"]
    if isinstance(tool_to_use, str):
        return tool_to_use
    args = task["args"]
    try:
        if isinstance(args, str):
            resolved_args = _resolve_arg(args, observations)
        elif isinstance(args, dict):
            resolved_args = {
                key: _resolve_arg(val, observations) for key, val in args.items()
            }
        else:
            # This will likely fail
            resolved_args = args
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}.)"
            f" Args could not be resolved. Error: {repr(e)}"
        )
    try:
        return tool_to_use.invoke(resolved_args, config)
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}."
            + f" Args resolved to {resolved_args}. Error: {repr(e)})"
        )


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
    # $1 or ${1} -> 1
    ID_PATTERN = r"\$\{?(\d+)\}?"

    def replace_match(match):
        # If the string is ${123}, match.group(0) is ${123}, and match.group(1) is 123.

        # Return the match group, in this case the index, from the string. This is the index
        # number we get back.
        idx = int(match.group(1))
        return str(observations.get(idx, match.group(0)))

    # For dependencies on other tasks
    if isinstance(arg, str):
        return re.sub(ID_PATTERN, replace_match, arg)
    elif isinstance(arg, list):
        return [_resolve_arg(a, observations) for a in arg]
    else:
        return str(arg)


@as_runnable
def schedule_task(task_inputs, config):
    task: Task = task_inputs["task"]
    observations: Dict[int, Any] = task_inputs["observations"]
    try:
        observation = _execute_task(task, observations, config)
    except Exception:
        import traceback

        observation = traceback.format_exception()  # repr(e) +
    observations[task["idx"]] = observation


def schedule_pending_task(
    task: Task, observations: Dict[int, Any], retry_after: float = 0.2
):
    while True:
        deps = task["dependencies"]
        if deps and (any([dep not in observations for dep in deps])):
            # Dependencies not yet satisfied
            time.sleep(retry_after)
            continue
        schedule_task.invoke({"task": task, "observations": observations})
        break


@as_runnable
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    """Group the tasks into a DAG schedule."""
    # For streaming, we are making a few simplifying assumption:
    # 1. The LLM does not create cyclic dependencies
    # 2. That the LLM will not generate tasks with future deps
    # If this ceases to be a good assumption, you can either
    # adjust to do a proper topological sort (not-stream)
    # or use a more complicated data structure
    tasks = scheduler_input["tasks"]
    args_for_tasks = {}
    messages = scheduler_input["messages"]
    # If we are re-planning, we may have calls that depend on previous
    # plans. Start with those.
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)
    # ^^ We assume each task inserts a different key above to
    # avoid race conditions...
    futures = []
    retry_after = 0.25  # Retry every quarter second
    with ThreadPoolExecutor() as executor:
        for task in tasks:
            deps = task["dependencies"]
            task_names[task["idx"]] = (
                task["tool"] if isinstance(task["tool"], str) else task["tool"].name
            )
            args_for_tasks[task["idx"]] = task["args"]
            if (
                # Depends on other tasks
                deps
                and (any([dep not in observations for dep in deps]))
            ):
                futures.append(
                    executor.submit(
                        schedule_pending_task, task, observations, retry_after
                    )
                )
            else:
                # No deps or all deps satisfied
                # can schedule now
                schedule_task.invoke(dict(task=task, observations=observations))
                # futures.append(executor.submit(schedule_task.invoke dict(task=task, observations=observations)))

        # All tasks have been submitted or enqueued
        # Wait for them to complete
        wait(futures)
    # Convert observations to new tool messages to add to the state
    new_observations = {
        k: (task_names[k], args_for_tasks[k], observations[k])
        for k in sorted(observations.keys() - originals)
    }
    tool_messages = [
        FunctionMessage(
            name=name, content=str(obs), additional_kwargs={"idx": k, "args": task_args}, tool_call_id = k
        )
        for k, (name, task_args, obs) in new_observations.items()
    ]
    return tool_messages

@as_runnable
def plan_and_schedule(state):
    # Made below change from original example. This is called inside this function here whereas in example code it is outside. 
    planner = create_planner(llm, tools, prompt)
    messages = state["messages"]
    tasks = planner.stream(messages)
    print("TASKS PRINT = {}".format(tasks))
    # Begin executing the planner immediately
    try:
        tasks = itertools.chain([next(tasks)], tasks)
    except StopIteration:
        # Handle the case where tasks is empty.
        print("TASKS IS EMPTY")
        tasks = iter([])
    scheduled_tasks = schedule_tasks.invoke(
        {
            "messages": messages,
            "tasks": tasks,
        }
    )
    return {"messages": scheduled_tasks}

# Joiner Component
class FinalResponse(BaseModel):
    """The final response/answer."""

    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""

    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]


def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    response = [AIMessage(content=f"Thought: {decision.thought}")]
    if isinstance(decision.action, Replan):
        return {"messages": response + [
            SystemMessage(
                content=f"Context from last attempt: {decision.action.feedback}"
            )
        ]
        }
    else:
        return {"messages": response + [AIMessage(content=decision.action.response)]}


def select_recent_messages(state) -> dict:
    messages = state["messages"]
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage):
            break
    return {"messages": selected[::-1]}


# Main function.
if __name__ == '__main__':
    
    _get_pass("OPENAI_API_KEY")
    tools = tools()
    prompt = planner()

    llm = ChatOpenAI(model="gpt-4-turbo-preview")
    # This is the primary "agent" in our application
    # planner = create_planner(llm, tools, prompt)
    example_question = "What's the temp of sf + 5?"

    
    # Task Fetching unit invocation
    # This is where the code seems to hang
    tool_messages = plan_and_schedule.invoke({"messages": [HumanMessage(content=example_question)]})['messages']
    print("Tool Messages: {}".format(tool_messages))

    print("------------------------Joiner Component---------------------------------")
   

    joiner_prompt = joiner()
    llm = ChatOpenAI(model="gpt-4-turbo-preview")

    runnable = joiner_prompt | llm.with_structured_output(JoinOutputs)
    joiner = select_recent_messages | runnable | _parse_joiner_output
    input_messages = [HumanMessage(content=example_question)] + tool_messages
    joiner.invoke({"messages": input_messages})
# %%

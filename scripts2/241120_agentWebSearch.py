# https://www.kaggle.com/code/marcinrutecki/llm-enhanced-web-search-the-tavily-lang-chain

import os
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


from langchain import hub
instructions = """You are an experienced researcher who always finds high-quality and relevant information on the Internet."""

base_prompt = hub.pull("langchain-ai/openai-functions-template")

prompt = base_prompt.partial(instructions=instructions)
print(prompt)


# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model_name="gpt-4", temperature=0, api_key=OPENAI_API_KEY)
from langchain_huggingface import HuggingFaceEndpoint
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
    repo_id = repo_id, max_new_tokens = 250, temperature = 0
)


from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
api_wrapper = TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY)
tavily_tool = TavilySearchResults(api_wrapper=api_wrapper)

tools = [tavily_tool]


from langchain.agents import AgentExecutor, create_openai_functions_agent
agent = create_openai_functions_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = "使用中文帮助生成 抑郁症 症状和治愈 建议方案。以问答对方式描述，然后，再从主要症状展开，给出更详细的治愈 建议方案"
result = agent_executor.invoke({"input": query})
print(result['output'])

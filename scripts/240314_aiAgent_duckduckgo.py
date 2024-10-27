# https://dev.to/timesurgelabs/how-to-make-an-ai-agent-in-10-minutes-with-langchain-3i2n
# pip install -U duckduckgo-search

import requests
from bs4 import BeautifulSoup


# defining Headers for Web Req
# ddg_search = DuckDuckGoSearchResults()
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
}


# parsing HTML content
def parse_html(content) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    text_content_with_links = soup.get_text()
    return text_content_with_links


# fetching webpage content
def fetch_web_page(url: str) -> str:
    response = requests.get(url, headers=HEADERS)
    return parse_html(response.content)


from langchain_community.tools import Tool, DuckDuckGoSearchResults
# creating the webFetcher tool
web_fetch_tool = Tool.from_function(
    func=fetch_web_page,
    name="WebFetcher",
    description="Fetches the content of a web page"
)


# setting up the summarizer
from langchain.prompts import PromptTemplate
prompts_template = "Summarize the following content: {content}"

# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo-16k")

# from langchain_community.llms import HuggingFaceEndpoint
# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# llm = HuggingFaceEndpoint(
#     repo_id=repo_id, max_new_tokens=250, temperature=0.5)

from langchain_community.llms import Ollama
# llm = Ollama(model="gemma:2b")
llm = Ollama(model="mistral")


from langchain.chains import LLMChain
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompts_template)
)

summarize_tool = Tool.from_function(
    func=llm_chain.run,
    name="Summarizer",
    description="Summarizes a web page"
)


# init the agent
from langchain.agents import initialize_agent, AgentType
tools = [DuckDuckGoSearchResults(), web_fetch_tool, summarize_tool]
agent = initialize_agent(
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    # handle_parsing_errors=True,
    verbose=True
)


# running the agent
prompt = "Research how to use the requests library in Python. Use your tools to search and summarize content into a guide on how to use the requests library."

print(agent.run(prompt))

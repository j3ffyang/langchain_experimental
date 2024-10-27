# https://stackoverflow.com/questions/77161157/langchain-not-working-with-transformers-models-how-to-use-langchain-transfor

from langchain_community.llms import HuggingFacePipeline
from langchain_community.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
# from langchain.schema import 

tools = [
    Tool(
        name="Music Search - Bo",
        func=lambda x: "'All I Want For Christmas Is You' by Mariah Carey.", # Mocked func
        description="A music search engine. Use this more than the normal search if the question is about Music, like 'who is the singer of singer of yesterday?' or 'what is the most popular song in 2022?'",
    ),
]

model_load = 'distilgpt2'   # downloaded locally
llm_huggingface = HuggingFacePipeline.from_model_id(
    model_id=model_load,
    task="text-generation",
    model_kwargs={"max_length": 500},
    # pipe.tokenizer.pad_token_id = model.config.eos_token_id,
)
llm_openai = OpenAI(temperature=0.1)

agent = initialize_agent(
    tools,
    # llm_openai,
    llm_huggingface,  # doesn't work yet
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
# print(agent)

answer = agent.run("what is the most famous song of Christmas")
print(answer)

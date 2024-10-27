import os
from langchain.prompts import PromptTemplate
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.chains import LLMChain

# os.environ['HUGGINGFACEHUB_API_TOKEN'] = 'HF_API_KEY'
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


template = """Question: {question}

Answer: """
prompt = PromptTemplate(template=template, input_variables=['question'])

# user question
question = "Which NFL team won the Super Bowl in the 2010 season?"

# # initialize Hub LLM
hub_llm = HuggingFaceHub(
    repo_id='google/flan-t5-xxl',
    model_kwargs={"temperature": 0.5, "max_length": 64}
)


# create prompt template > LLM chain
llm_chain = LLMChain(prompt=prompt, llm=hub_llm)


# ask the user question about NFL 2010
print(llm_chain.run(question))

# # qs = [
#     {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
#     {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
#     {'question': "Who was the 12th person on the moon?"},
#     {'question': "How many eyes does a blade of grass have?"}
# ]
# res = llm_chain.generate(qs)
# res

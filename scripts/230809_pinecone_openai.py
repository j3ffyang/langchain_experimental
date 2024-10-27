from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
# import langchain


davinci = OpenAI(model_name='text-davinci-003')

template = """Question: {question}

Answer: """

prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

llm_chain = LLMChain(
    prompt=prompt,
    # prompt=long_prompt,
    llm=davinci
)

# # user question
# question = "Which NFL team won the Super Bowl in the 2010 season?"

# print(llm_chain.run(question))

qs = [
    {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
    {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
    {'question': "Who was the 12th person on the moon?"},
    {'question': "How many eyes does a blade of grass have?"}
]
print(type(llm_chain.generate(qs)))
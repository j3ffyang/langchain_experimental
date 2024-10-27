from langchain import FewShotPromptTemplate
from langchain import PromptTemplate
from langchain import OpenAI

openai = OpenAI(
    model_name="text-davinci-003",
)

examples = [ 
    {
        "query": "How are you?",
        "answer": "I can't complain but sometimes I still do."
    }, {
        "query": "What time is it?",
        "answer": "It's 12:00."
    }
]

# create an example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions!
prefix = """The following are exeprpts from conversations with an AI assistant. The assistant is typically sarcastic and witty, producing creative and funny responses to the users questions. Here are some examples in Chinese language:
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

query = "What is the meaning of life?"
print(few_shot_prompt_template.format(query=query))

print()
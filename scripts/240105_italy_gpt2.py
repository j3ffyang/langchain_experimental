from langchain.llms.huggingface_pipeline import HuggingFacePipeline
# cohere = Cohere(model='command-xlarge')
microsoft = "microsoft/DialoGPT-medium"
gpt2 = "gpt2"

hf = HuggingFacePipeline.from_model_id(
    model_id=gpt2,
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 200,
                     "pad_token_id": 50256}
)

from langchain.prompts import PromptTemplate
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

# question = "Does the money buy happiness?"
question = "Tell me about Europe history?"

from langchain.globals import set_verbose, set_debug
set_debug(True)
set_verbose(True)

print(chain.invoke({"question": question}))

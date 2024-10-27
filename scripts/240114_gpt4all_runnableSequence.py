# https://python.langchain.com/docs/integrations/providers/gpt4all
# https://python.langchain.com/docs/integrations/llms/gpt4all

from langchain.prompts import PromptTemplate
template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])


from langchain_community.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
callbacks = [StreamingStdOutCallbackHandler()]
# Instantiate the model. Callbacks support token-wise streaming
llm = GPT4All(
    model="/home/jeff/.cache/gpt4all/mistral-7b-openorca.gguf2.Q4_0.gguf",
    device='gpu', n_threads=8,
    callbacks=callbacks, verbose=True)

# response = llm("Once upon a time, ", max_tokens=3)
# print(response)


# from langchain.chains import LLMChain
# llm_chain = LLMChain(prompt=prompt, llm=llm)
from langchain_core.runnables import RunnableSequence
llm_chain = RunnableSequence(prompt | llm)
llm_chain.invoke("Tell me about Italian Pasta")

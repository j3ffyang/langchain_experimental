# https://www.youtube.com/watch?v=gLfPzCYo-VQ

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    # trust_remote_code=True,
    device_map="auto",
    max_length=200,
)


# prompt = "Tell me about Italy"
# sequences = pipeline(
#     prompt,
#     max_new_tokens=400,
#     # max_length=400,
#     do_sample=True,
#     top_k=1,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
# )
# print(sequences)


from langchain.prompts import PromptTemplate
template = """
You are an intelligent chatbot. Help the following question with brilliant answers.
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])


from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipeline,
                          model_kwargs={'temperature':0.01})


from langchain.chains import LLMChain
llm_chain = LLMChain(prompt=prompt, llm=llm)

# question = "Explain the history of Italy and its food"
question = "Tell me about Italy"
print(llm_chain.invoke(question))

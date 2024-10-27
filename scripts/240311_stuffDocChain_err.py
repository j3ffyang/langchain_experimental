# https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.stuff.StuffDocumentsChain.html

from langchain_core.prompts import PromptTemplate
document_prompt = PromptTemplate(
    input_variables=["page_content"],
    template="{page_content}"
)
document_variable_name = "context"
prompt = PromptTemplate.from_template(
    "Summarize this content: {context}"
)

from langchain_community.llms import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-xxl", max_new_tokens=512, temperature=0.5)

from langchain.chains import StuffDocumentsChain, LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt)
chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name
)

print(chain.invoke(input = "where is the harry potter's owl?"))

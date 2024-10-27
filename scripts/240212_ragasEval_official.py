# https://docs.ragas.io/en/v0.0.18/howtos/integrations/langchain.html

# from langchain.document_loaders import TextLoader
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://en.wikisource.org/wiki/Hans_Andersen%27s_Fairy_Tales/The_Emperor%27s_New_Clothes")

from langchain.indexes import VectorstoreIndexCreator
index = VectorstoreIndexCreator().from_loaders([loader])


from langchain_community.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
callbacks = [StreamingStdOutCallbackHandler()]
llm = GPT4All(
    model="/home/jeff/.cache/huggingface/hub/gpt4all/mistral-7b-openorca.Q4_0.gguf",
    device='gpu',
    n_threads=8)

# from langchain.chat_models import ChatOpenAI
# llm = ChatOpenAI(temperature=0)

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=index.vectorstore.as_retriever(),
    return_source_documents=True,
)

# question = "How did New York city get its name?"
# result = qa_chain({"query": question})
# print(result["result"])

eval_questions = [
    "What is the population of New York City as of 2020?",
    "Which borough of New York City has the highest population?",
    "What is the economic significance of New York City?",
    "How did New York City get its name?",
    "What is the significance of the Status of Liberty in New York City?",
]

eval_answers = [
    "8,804,190",
    "Brooklyn",
    "New York City's economic significance is vast, as it serves as the global financial capital, housing Wall Street and major financial institutions. Its diverse economy spans technology, media, healthcare, education, and more, making it resilient to economic fluctuations. NYC is a hub for international business, attracting global companies, and boasts a large, skilled labor force. Its real estate market, tourism, cultural industries, and educational institutions further fuel its economic prowess. The city's transportation network and global influence amplify its impact on the world stage, solidifying its status as a vital economic player and cultural epicenter.",
    "New York City got its name when it came under British control in 1664. King Charles II of England granted the lands to his brother, the Duke of York, who named the city New York in his own honor.",
    "The Statue of Liberty in New York City holds great significance as a symbol of the United States and its ideals of liberty and peace. It greeted millions of immigrants who arrived in the U.S. by ship in the late 19th and early 20th centuries, representing hope and freedom for those seeking a better life. It has since become an iconic landmark and a global symbol of cultural diversity and freedom.",
]

examples = [
    {"query": q, "ground_truths": [eval_answers[i]]}
    for i, q in enumerate(eval_questions)
]

# result = qa_chain({"query": eval_questions[1]})
# result = qa_chain(examples[2])
# print(result["result"])


from ragas.langchain.evalchain import RagasEvaluatorChain
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# create evaluation chains
# faithfulness_chain = RagasEvaluatorChain(metric=faithfulness)
# answer_rel_chain = RagasEvaluatorChain(metric=answer_relevancy)
# context_rel_chain = RagasEvaluatorChain(metric=context_precision)
# context_recall_chain = RagasEvaluatorChain(metric=context_recall)

result = evaluate(
    amnesty_qa["eval"].select(range(10)),  # showing only 10 for demonstration
    metrics=[faithfulness],
    llm=llm,
)

print(result)

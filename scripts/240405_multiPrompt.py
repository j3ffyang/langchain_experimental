# https://stackoverflow.com/questions/78280330/langchain-sequential-chaining-with-lcel

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

context = "I run a blog; I address a semi-technical audience on LinkedIn; my writing style uses commonly understandable language to make it sound like a personal blog post using anecdotes and jokes while keeping an overall professional tone"
goal = "be as creative as possible; take my initial ideas and augment, optimize given the context above; function as my creative writing assistant"
topic = "Data & AI"
thoughts = "Data & AI related stuff"

structure_prompt = PromptTemplate.from_template(
    """You are a writer. Given the following insights you are tasked with brainstorming a structure for a new blog post; be creative, add things I might have not considered and build a structure for an informative yet engaging blog post
    Context: {context}
    Goal: {goal}
    Topic: {topic}
    Thoughts: {thoughts}
    """
)

review_prompt = PromptTemplate.from_template(
    """You are an expert reviewer. Given this drafted structure for a blog post, what would you optimize; what would you add, focus or remove given the context, topic and thoughts explained above:
    Structure: {structure}
    Provide concrete feedback in a cohesive and comprehensive form that a writer can optimize the original sturcture accordingly.
    """
)

optimization_prompt = PromptTemplate.from_template(
    """You are a writer. Optimize the following blog post structure given the feedback review provided.
    Structure: {structure}
    Review: {review}
    """
)

llm = ChatOpenAI()

structure_chain = structure_prompt | llm | StrOutputParser()
review_chain = review_prompt | llm | StrOutputParser()
optimization_chain = optimization_prompt | llm | StrOutputParser()

chain = ({"structure" : structure_chain}
        | RunnablePassthrough.assign(review=review_chain)
        | RunnablePassthrough.assign(optimization=optimization_chain))

result = chain.invoke({"context": context, "goal" : goal, "topic" : topic, "thoughts" : thoughts})
print(result)

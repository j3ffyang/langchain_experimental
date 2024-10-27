# https://stackoverflow.com/questions/76385146/openai-api-error-why-do-i-still-get-the-module-openai-has-no-attribute-chat

import os
import openai

# openai.api_key = ""

prompt = f"""
write a short story about a person who is going to a party.
"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt},
    ],
    temperature=0,
    max_tokens=2024,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
)

print(response["choices"][0]["message"]["content"]) 

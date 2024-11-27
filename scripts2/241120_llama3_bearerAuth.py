import requests
import os
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

AUTH_TOKEN_STRING = "Bearer "+HUGGINGFACEHUB_API_TOKEN

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B"
# headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}
headers = {"Authorization": AUTH_TOKEN_STRING}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

output = query({
	"inputs": "Can you please let us know more details about your ",
})

print(output)

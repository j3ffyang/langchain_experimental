# %%
from langflow import load_flow_from_json
TWEAKS = {
  "SupabaseVectorStore-fJXrJ": {},
  "HuggingFaceEmbeddings-c1lXf": {},
  "WebBaseLoader-0TPwO": {}
}
flow = load_flow_from_json("240514_kb_semantic.json", tweaks=TWEAKS)
# Now you can use it like any chain
inputs = {"input": "message"}
flow(inputs)

# %%

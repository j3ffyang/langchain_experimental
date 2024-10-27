# https://www.markhneedham.com/blog/2023/06/23/hugging-face-run-llm-model-locally-laptop/

from huggingface_hub import hf_hub_download

model_id = "lmsys/fastchat-t5-3b-v1.0"
filenames = [
    "pytorch_model.bin", "added_tokens.json", "config.json", "generation_config.json",
    "special_tokens_map.json", "spiece.model", "tokenizer_config.json"
]

for filename in filenames:
    downloaded_model_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
    )

    print(downloaded_model_path)

print(downloaded_model_path)

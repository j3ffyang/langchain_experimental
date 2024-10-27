import re

pattern = r'问题：(.+?)\n\n答案：(.+?)\n'

with open('./qa_jsonl_all.txt', 'r') as f:
    text = f.read()

update_text = re.sub(pattern, '', text)
print(update_text)

# with open('./qa_jsonl_all.json', 'w') as f:
#     f.write(update_text)


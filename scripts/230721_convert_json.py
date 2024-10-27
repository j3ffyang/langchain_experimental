import json
import re

with open('qa_jsonl_all.txt', 'r') as f:
    text = f.read()

# pattern = r'问题：(.+?)\n\n答案：(.+?)\n\n'
pattern = r'问题：(.+?)\n\n答案：(.+?)\n'
matches = re.findall(pattern, text)

data = []
for match in matches:
    question = match[0]
    answer = match[1]
    data.append({
        "question": question,
        "answer": answer
    })

with open('qa_converted.json', 'w') as f:
     json.dump(data, f, ensure_ascii=False, indent=4)

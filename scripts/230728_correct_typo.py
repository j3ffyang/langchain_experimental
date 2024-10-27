import json
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str, required=True, default="/home/jeff/Downloads/scratch/instguid.git/gpt/hlmgpt/jeff_tutorials/data/ximalaya/original_md")
parser.add_argument("-o", "--output_dir", type=str, required=True, default="/home/jeff/Downloads/scratch/instguid.git/gpt/hlmgpt/scripts/jeff_tutorials/data/ximalaya/ximalaya_typo_corrected")
args = parser.parse_args()


items = []
with open("typo_corrected.json", "r") as read_file:
    data = json.load(read_file)
    items = data["words"]


def correction(read_file):
    """
    :param read_file:
    :return: correction
    """
    content = read_file.read()
    for item in items:
        for typo in item["typo"]:
            content = content.replace(typo, item["corrected"])
    return content


os.makedirs(f'{args.output_dir}', exist_ok=True)

for filename in os.listdir(f'{args.input_dir}'):
    f_in  = os.path.join(f'{args.input_dir}',  filename)
    f_out = os.path.join(f'{args.output_dir}', filename)
    with open(f_in, "r") as read_file:
        new_content = correction(read_file)
    with open(f_out, "w") as write_file:
        write_file.write(new_content)
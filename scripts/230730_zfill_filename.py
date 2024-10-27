# PYTHON_ARGCOMPLETE_OK
import os
import shutil
import argparse
import argcomplete

parser = argparse.ArgumentParser()
parser.add_argument('--src_dir', type=str, default='/home/jeff/Downloads/scratch/instguid.git/gpt/hlmgpt/jeff_tutorials/data/ximalaya/original_md')
parser.add_argument('--dst_dir', type=str, default='/home/jeff/Downloads/scratch/instguid.git/gpt/hlmgpt/jeff_tutorials/data/ximalaya/original_md_renamed')
argcomplete.autocomplete(parser)
args = parser.parse_args()


src_dir = args.src_dir
dst_dir = args.dst_dir

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
    print(f"Created directory {dst_dir}")

files = os.listdir(src_dir)
for file in files:
    if file.endswith('_笔记.md'):
        num = file.split('-')[0]
        new_name = f"{num.zfill(3)}-{file.split('-')[1]}"
        # print(f"renamed '{file}' to '{new_name}'")
        # Cautious that the original dir will be replaced
        # os.rename(os.path.join(dir_path, file), os.path.join(dir_path, new_name))
        # shutil.move(os.path.join(src_dir, file), os.path.join(dst_dir, new_name))
        shutil.copy2(os.path.join(src_dir, file), os.path.join(dst_dir, new_name))


if __name__ == "__main__":
    parser.parse_args()
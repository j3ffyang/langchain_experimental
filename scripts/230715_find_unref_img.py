## https://www.perplexity.ai/search/51bd934f-6fc3-416d-af04-899b8961aff4?s=u

import os
import glob 


file_dir = "/home/jeff/Downloads/scratch/instguid.git/assets"
md_dir = "/home/jeff/Downloads/scratch/instguid.git"

file_list = []
for file_path in glob.glob(os.path.join(file_dir, '*')):
    filename = os.path.basename(file_path)
    file_list.append(filename)
    # print(file_list)


referenced_files = []
for dir_name, subdir_list, file_list in os.walk(md_dir):
    for file_path in glob.glob(os.path.join(dir_name, '*.md')):
    # for file_path in glob.glob(os.path.join(md_dir, '*.md')):
        with open(file_path, 'r') as f:
            content = f.read()
            for filename in file_list:
                if filename in content:
                    referenced_files.append(filename)
                    break
# print(referenced_files)
unreferenced_files = [filename for filename in file_list if filename not in referenced_files]
print(unreferenced_files)
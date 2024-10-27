# https://www.phind.com/search?cache=xose1mft0h1uynqspu12nkr9

import os
import glob

directory = './'

pattern = "^20*.md"
# pattern = '2023*.py'
file_list = glob.glob(os.path.join(directory, pattern))

for file_path in file_list:
    print(file_path)

for file_path in file_list:
    base_name = os.path.basename(file_path)
    new_base_name = base_name.replace('20', '', 1)
    new_file_path = os.path.join(directory, new_base_name)
    os.rename(file_path, new_file_path)


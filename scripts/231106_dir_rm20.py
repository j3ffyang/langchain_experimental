import os

parent_dir = "./"

for root, dirs, files in os.walk(parent_dir, topdown=False):
    for name in dirs:
        if name.startswith("20"):
            new_name = name[2:]
            os.rename(os.path.join(root, name), os.path.join(root, new_name))

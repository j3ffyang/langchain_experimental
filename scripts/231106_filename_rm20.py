import os

# Specify the directory path
directory = './'

# List all files in the directory
files = os.listdir(directory)

# Iterate over the files and rename them
for filename in files:
    if filename.startswith('20') and filename.endswith('.md'):
        new_filename = filename[2:4] + filename[4:]
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))


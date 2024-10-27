import os 

# Specify the dir path
directory = '/home/jeff/Downloads/scratch/instguid.git/gpt/hlmgpt/data/ximalaya/original_md'

# Create a list to store the extracted substrings
substrings = []


for filename in os.listdir(directory):
    # Check if the filename containers '-' and '_笔记.md'
    if '-' in filename and '_笔记.md' in filename:
        # Get the substring between them
        q_in_subj = filename.split('-')[1].split('_笔记.md')[0]
        file_serial = "{:05d}".format(int(filename.split('-')[0]) + 1)
        # print(q_in_subj)

        content = ""

    # if os.path.isfile(os.path.join(directory, filename)):
        with open(os.path.join(directory, filename), 'r') as file:
            content = file.read()   # or file.readlines() for line-by-line
            # Perform operations on the file content
        
        # print(content)

        # Specify the output file path
        output_file = f"/tmp/{file_serial}_{q_in_subj}.txt"


        # Open the output file in write mode
        with open(output_file, 'w') as file:
            file.write(content)


print("Content saved successfully!")
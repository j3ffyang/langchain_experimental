import os 
import glob

img_dir = "/home/jeff/Downloads/scratch/instguid.git/assets"
md_dir = "/home/jeff/pool/dyn/Downloads/scratch/instguid.git"

# img_dir = "/tmp/haoqi/img_dir"
# md_dir = "/tmp/haoqi/md_dir"

img_list = []

for file_path in glob.glob(os.path.join(img_dir, "*")):
    filename = os.path.basename(file_path)
    # print(filename)
    img_list.append(filename)
    # print(img_list)


referenced_files = []

for dir_name, subdir_list, file_list in os.walk(md_dir):
    for file_path in glob.glob(os.path.join(dir_name, '*.md')):
        with open(file_path, 'r') as f:
            content = f.read()
            for filename in img_list:
                if filename in content:
                    referenced_files.append(filename)
                    # if filename not in content:
                    #     print(f"{filename} is NOT referenced in {file_path}")

# set = dedup'd
referenced_files = list(set(referenced_files))
unreferenced_files = [element for element in img_list if element not in referenced_files]
print(len(unreferenced_files))
print(unreferenced_files)

# print("Referenced files:")
print(len(referenced_files))
# print(set(referenced_files))
# for filename in referenced_files:
#     print(filename)

###

# Deduplicated list: [1, 2, 3, 4, 5]
# original_list = [1, 2, 2, 3, 4, 4, 5]
# deduplicated_list = []

# for item in original_list:
#     if item not in deduplicated_list:
#         deduplicated_list.append(item)

# print("Deduplicated list:", deduplicated_list)
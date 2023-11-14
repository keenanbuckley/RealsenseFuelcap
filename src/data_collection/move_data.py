import os, shutil
import json

# src_path = "temp_data"
# dest_path = "data"

# with open("data/new_data.json", 'r') as f: 
#     data = json.load(f)

# files = list(data.keys())

# for file in files:
#     shutil.copy(f"{src_path}/color/{file}.png", f"{dest_path}/color/{file}.png")
#     shutil.copy(f"{src_path}/depth/{file}.npy", f"{dest_path}/depth/{file}.npy")

    
    
with open("data/new_data.json", 'r') as f: 
    data = json.load(f)
files = list(data.keys())


drive_data = {}
for file in files[:50]:
    drive_data[file] = data[file]
    shutil.copy(f"data/depth/{file}.npy", f"drive_data/depth/{file}.npy")
    shutil.copy(f"data/color/{file}.png", f"drive_data/color/{file}.png")


json_str = json.dumps(drive_data, indent=4).replace('[\n\t','[').replace(",\n            ", ", ").replace("[\n            ", '[').replace("\n        ]", "]")
with open("drive_data/data.json", 'w') as f:
    f.write(json_str)

# for file in files:
#     if f"{file}.png" not in os.listdir("data/color"):
#         print(f"{file}.png not in data")
#     if f"{file}.npy" not in os.listdir("data/depth"):
#         print(f"{file}.npy not in data")

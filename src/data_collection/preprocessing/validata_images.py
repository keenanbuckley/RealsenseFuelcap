import json, os
color_image_path = "data/color"

with open("data/train_data.json") as f:
    data = json.load(f)

json_files = data.keys()
img_files = os.listdir(color_image_path)


for file in json_files:
    if f"{file}.png" not in img_files:
        print(f"{file} not not in color data")

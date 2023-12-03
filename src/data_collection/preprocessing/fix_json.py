import json, random, os, sys


def get_fixed_data(filepath: str, filename: str) -> dict:
    if "RealWorldBboxData" in filepath:
        with open("NewBboxData/RealWorldBboxData/all_data.json") as f:
            data = dict(json.load(f))
        fixed_data = data
        for file in fixed_data.keys():
            fixed_data[file]["img_dir"] = "NewBboxData/RealWorldBboxData"
        return fixed_data
    

    with open(f"{filepath}/{filename}", 'r') as file:
        data = dict(json.load(file))
    


    fixed_data = {}
    for file, file_data in data.items():
        fixed_file_data = {}

        keypoints = file_data["keypoints"]
        fixed_keypoints = []
        for kp in keypoints:
            x = round(kp["pos"]["x"])
            y = 720-round(kp["pos"]["y"])
            i = int(kp["name"][-1])+1
            fixed_keypoints.append([x,y,i])
        fixed_file_data["keypoints"] = fixed_keypoints

        bbox = file_data["bbox"]
        fbb = [
            bbox["topLeft"]["x"],
            720-bbox["topLeft"]["y"],
            bbox["bottomRight"]["x"],
            720-bbox["bottomRight"]["y"]
        ]

        fixed_bbox = [min(fbb[0], fbb[2]), min(fbb[1], fbb[3]), max(fbb[0], fbb[2]), max(fbb[1], fbb[3])]
        fixed_file_data["bbox"] = fixed_bbox


        fixed_file_data["cameraData"] = file_data["cameraData"]
        fixed_file_data["fuelCapData"] = file_data["fuelCapData"]
        fixed_file_data["img_dir"] = filepath

        fixed_data[file] = fixed_file_data

        
    return fixed_data

def create_train_test(data: dict, ratio: float) -> list[dict, dict]:
    files = list(data.keys())

    if ratio >= 1 or ratio <= 0:
        raise ValueError("Ratio must be between 0 and 1 exclusive", ratio)

    num_train = int(len(files) * ratio)

    train_files = random.sample(files, num_train)
    test_files = [f for f in files if f not in train_files]

    train_data = {f: data[f] for f in train_files}
    test_data = {f: data[f] for f in test_files}

    if len(train_data.keys()) + len(test_data.keys()) != len(data.keys()):
        raise ValueError("Train and Test data do not add up to original")
    
    return [train_data, test_data]

def read_data(filepath, filename) -> dict:
    with open(f"{filepath}/{filename}", 'r') as f:
        return json.load(f)

def write_data(filepath, filename, fixed_data):
    data_str = json.dumps(fixed_data, indent=4).replace("[\n                ", "[").replace(",\n                ", ", ").replace("\n            ]", "]")
    data_str = data_str.replace(": [\n            ", ": [").replace(",\n            ", ", ").replace("\n        ],", "],")
    with open(f"{filepath}/{filename}", 'w') as f:
        f.write(data_str)

def fix_all_directories(folder_path, filename, directories):
    for dir in directories:
        filepath = f"{folder_path}/{dir}"
        fixed_data = get_fixed_data(filepath, filename)
        write_data(filepath, "frame_data.json", fixed_data)

def assemble_data(folder_path, directories) -> dict:
    filename = "frame_data.json"
    all_data = {}
    for dir in directories:
        dir_data = read_data(f"{folder_path}/{dir}", filename)

        for k,v in dir_data.items():
            all_data[k] = v

    return all_data

if __name__ == "__main__":
    sys.argv

    # Specify the path to the folder
    folder_path = 'NewBboxData'
    filename = "FrameData.json"

    if len(sys.argv) == 3:
        folder_path = sys.argv[1]
        filename = sys.argv[2]

    print(folder_path, filename)
    
    # Get a list of all directories in the folder
    directories = [entry.name for entry in os.scandir(folder_path) if entry.is_dir()]
    directories = [d for d in directories if d not in ["accumulated_data"]]

    fix_all_directories(folder_path, filename, directories)

    all_data = assemble_data(folder_path, directories)

    filepath = f"{folder_path}/accumulated_data"
    write_data(filepath, "frame_data.json", all_data)

    train_data, test_data = create_train_test(all_data, 0.8)

    write_data(filepath, "frame_data_train.json", train_data)
    write_data(filepath, "frame_data_test.json", test_data)

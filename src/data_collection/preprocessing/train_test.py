import sys, json, random

def write_data(path, data, files, datatype):
    sub_data = dict()
    for file in files:
        sub_data[file] = data[file]
    with open(f"{path}/{datatype}_data.json", 'w') as file:
        json.dump(sub_data, file, indent=2)

if __name__ == "__main__":
    assert len(sys.argv) == 4
    path = sys.argv[1]
    source_file = sys.argv[2]
    proportion = float(sys.argv[3])

    print(path, source_file, proportion)

    source_file = f"{path}/{source_file}"

    with open(source_file, 'r') as file:
        data = dict(json.load(file))
    
    images = list(data.keys())

    settype = "train"
    print(f"Creating train set of {round(proportion*100)}% of dataset")
    train = random.sample(images, int(proportion * len(images)))
    write_data(path, data, train, "train")

    print(f"Creating test set of {round((1-proportion)*100)}% of dataset")
    test = [i for i in images if i not in train]
    write_data(path, data, test, "test")




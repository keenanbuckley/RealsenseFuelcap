
import json, random
def test_files():
    with open("src/data_collection/data_reading/combined_data/new_data.json", 'r') as file:
        data = json.load(file)

    with open("/home/david/stratom/RealsenseFuelcap/data/train_data.json",'r') as f:
        train_data = json.load(f)

        
    with open("/home/david/stratom/RealsenseFuelcap/data/test_data.json",'r') as f:
        test_data = json.load(f)

    all_files = data.keys()
    all_train = train_data.keys()
    all_test = test_data.keys()

    assert len(all_files) == len(all_test) + len(all_train)
    for file in all_train:
        assert file not in all_test
    for file in all_test:
        assert file not in all_train
        
def main():
    with open("src/data_collection/data_reading/combined_data/new_data.json", 'r') as file:
        data = json.load(file)
    

    files = list(data.keys())
    percentage = 0.8
    num_to_train = int(len(files) * percentage)

    train_list = random.sample(files, num_to_train)
    test_list = [i for i in files if i not in train_list]

    train_json = {i: data[i] for i in train_list}
    test_json = {i: data[i] for i in test_list}

    train_str = json.dumps(train_json, indent=4).replace('[\n\t','[').replace(",\n            ", ", ").replace("[\n            ", '[').replace("\n        ]", "]")
    with open("/home/david/stratom/RealsenseFuelcap/data/train_data.json",'w') as f:
        f.write(train_str)

        
    test_str = json.dumps(test_json, indent=4).replace('[\n\t','[').replace(",\n            ", ", ").replace("[\n            ", '[').replace("\n        ]", "]")
    with open("/home/david/stratom/RealsenseFuelcap/data/test_data.json",'w') as f:
        f.write(test_str)
    
    test_files()

if __name__ == "__main__":
    main()
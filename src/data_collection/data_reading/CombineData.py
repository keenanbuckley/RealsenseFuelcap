import json
import os


def Combine(main, new):
    outputfiles = os.path.expanduser('~/mines_ws/src/data_collection/data_reading/output/')  
    combined_data = os.path.expanduser('~/mines_ws/src/data_collection/data_reading/combined_data/') 

    with open(combined_data+main, "r") as file:
        original_content = json.load(file)
    with open(outputfiles+new, "r") as file:
        new_content = json.load(file)
    combined = {**original_content, **new_content}

    with open(combined_data + main, 'w') as json_file:
        json.dump(combined, json_file, indent=4)
Combine("main.json","outputKeenan.json")
Combine("main.json","outputDavid.json")
Combine("main.json","outputDavid2.json")
Combine("main.json","outputJaxon.json")
Combine("main.json","outputJaxon2.json")
Combine("main.json","outputXavier.json")
Combine("main.json","outputXavier2.json")
Combine("main.json","outputXavier3.json")




    


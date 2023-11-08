import json
import os

def OrganizeData(filename):
    # Open the specified JSON file from the 'Input' directory.
    inputDir = os.path.expanduser('~/mines_ws/src/data_collection/data_reading/Input/')  # Replace with your source directory path
    with open(inputDir + filename, "r") as file:
        data_content = json.load(file)
    
    # Initialize an empty dictionary to store organized data.
    my_dict = {}
    copycount = 0
    # Iterate over each item in the "items" list within the JSON data.
    for i in range(len(data_content["items"])):
        # Extract the ID of the current item as the key.
        key = data_content["items"][i]["id"]
        #Some were  copies of eachother :(
        if "copy" in key:
            continue
        # If the key is not already present in the dictionary, create a new sub-dictionary for it.
        if key not in my_dict:
            my_dict[key] = {}
        else:
            # If the key is already present, print an error message.
            print("Error: Keypoint already in dict")

        # Iterate over each annotation within the current item.
        for j in range(len(data_content["items"][i]["annotations"])):
            # Check for the existence of "bbox" (bounding box) in the annotations and add them to the dictionary.
            if data_content["items"][i]["annotations"][j].get("bbox") is not None:
                bbox_value = data_content["items"][i]["annotations"][j]["bbox"]
                my_dict[key]["Boundingbox"] = []

                # Check if the bounding box has exactly 4 values, otherwise print an error message.
                if len(bbox_value) != 4:
                    print("Incorrect Amount of boundingBoxes, #BoundingBoxes: ", len(bbox_value), " Key:", key, " " + filename)
                    continue
                my_dict[key]["Boundingbox"].append(bbox_value)

            # Check for the existence of "points" in the annotations and add them to the dictionary.
            if data_content["items"][i]["annotations"][j].get("points") is not None:
                points_value = data_content["items"][i]["annotations"][j]["points"]
                my_dict[key]["Keypoint"] = []

                # Check if the points list has exactly 20 values, otherwise print an error message.
                if len(points_value) != 20:
                    print("Incorrect Amount of Keypoints, #Keypoints: ", len(points_value), " Key:", key, " " + filename)
                    continue
                my_dict[key]["Keypoint"].append(points_value)
    # Write the organized data into an output JSON file within the 'Output' directory.
    outputDir = os.path.expanduser('~/mines_ws/src/data_collection/data_reading/output/output') 
    with open(outputDir + filename, 'w') as json_file:
        json.dump(my_dict, json_file, indent=4)

# Call the Organize function for each specified JSON file.
OrganizeData("Keenan.json")
OrganizeData("Jaxon.json")
OrganizeData("Xavier.json")
OrganizeData("David.json")
OrganizeData("Xavier2.json")
OrganizeData("David2.json")
OrganizeData("Jaxon2.json")
OrganizeData("Xavier3.json")

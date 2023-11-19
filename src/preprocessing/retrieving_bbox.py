import json
import os
import numpy as np 
#Adds clean bbox data to this list
bboxPointList = []

combined_data_path = os.path.expanduser('~/mines_ws/src/data_collection/data_reading/combined_data/main.json') 

with open(combined_data_path, "r") as file:
    #reads in json
    bboxdata = json.load(file)
#retrieves the bounding box using its number identifier
def retrieve_bbox(imageNum):
    #Checks key exists
    if(bboxdata.get(imageNum) is None):
        print("Key ", imageNum, " does not exist")
        return
    #Checks bounding box at key exists
    if(bboxdata[imageNum].get("Boundingbox") is None):
        print("Boundingbox ", imageNum, " does not exist")
        return
    #retrieves point
    bboxPoints = bboxdata[imageNum]["Boundingbox"]
    #rounds numbers and turns them to ints
    bboxPointsClean = [int(round(num, 0)) for num in bboxPoints[0]]
    bboxPointList.append(bboxPointsClean)

retrieve_bbox("1699122638638")
retrieve_bbox("1699122990883")
bboxPointArray = np.array((bboxPointList))
print(bboxPointList)


import sys
from os.path import dirname
sys.path.append(f'{dirname(__file__)}/..')

from keypoint_model import keypoint_model
import cv2, numpy as np


dpp = lambda x, y: np.sqrt(np.sum((x - y) ** 2))

def test_model(image_path: str, data_imdir=False):
    import json

    if data_imdir:
        with open(f"{image_path}/all_data.json", 'r') as f:
            data = dict(json.load(f))   
    else:
        with open(f"{image_path}/all_data.json", 'r') as f:
            data = dict(json.load(f))

    image_names = list(data.keys())

    kpModel = keypoint_model()
    loss = []
    num_rejected = 0
    for i,iname in enumerate(image_names):
        print(f"{i} of {len(image_names)}", end="\r")
        iData = data[iname]
        bbox = iData["bbox"]
        if data_imdir:

            imdir = iData["img_dir"]
            img = cv2.imread(f"{imdir}/Image/{iname}.png")
        else:
            img = cv2.imread(f"{image_path}/color/{iname}.png")
        
        keypoints_pred = kpModel.predict(img, bbox)[:, :2]
        keypoints_known = reshape_points(iData)
        

        known_sorted = matchPoints(keypoints_known, keypoints_pred)
        if known_sorted is None:
            print(f"Rejecting {iname}")
            num_rejected += 1
            continue
        total_loss= dpp(keypoints_pred, known_sorted)
        loss.append(total_loss)

    return np.array(loss), num_rejected, len(image_names)

def reshape_points(iData):
    keypoints_known = iData["keypoints"]
    keypoints_known = np.array([[keypoints_known[2*i], keypoints_known[2*i+1]] for i in range(len(keypoints_known) // 2)])
    return keypoints_known
        # print(keypoints)

def matchPoints(known, predicted):
    if known.shape != predicted.shape:
        print("Known and predicted must be the same shape")
    known_sorted = np.zeros_like(known)
    indexes = []
    num_rejected = 0
    for i in range(known.shape[0]):
        pred_point = predicted[i, :]
        min_dist, min_idx = np.inf, 0
        for j in range(known.shape[0]):
            dist = dpp(pred_point, known[j, :])
            if min_dist > dist:
                min_idx = j
                min_dist = dist
        known_sorted[i, :] = known[min_idx, :]
        indexes.append(min_idx)

    if not len(indexes) == len(set(indexes)):
        return None

    return known_sorted


def analyze_model(path = "data/RealWorldBboxData", im_dir = False):
    
    loss, num_reject, length = test_model(path, im_dir)
    mean = np.mean(loss)
    std = np.std(loss)
    return num_reject,length,mean,std

if __name__ == "__main__":
    analyze_model(path="/media/david/Loser Drive/Keypoint_model/NewData/accumulated_data")
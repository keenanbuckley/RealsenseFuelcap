from __future__ import division
from __future__ import print_function




import torch
from torchvision import transforms
# from hourglass import hg
from PIL import Image
import os
#from os import join
from time import time
import numpy as np
import cv2 
from transformations import *


    
class keypoint_model:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform_list = [CropAndPad(out_size=(256, 256))]
        transform_list.append(ToTensor())
        transform_list.append(Normalize())

        self.transform = transforms.Compose(transform_list)

        self.model = torch.load('./models/keypoints_detection.pth')

        self.keypoints = None
        self.item = None
        self.model.eval()

    def predict(self, image, bbox):
        # convert to PIL image
        image = Image.fromarray(image)

        #initialize input
        self.item = {'image': image, 'bb': bbox}
        self.item = self.transform(self.item)

        img = self.item["image"].unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            keypoints_prediction = self.model(img)
        
        self.keypoints = np.array(keypoints_prediction[0].cpu())[0,:,:,:]
        
        x_coords = []
        y_coords = []
        for i in range(10):
            # Find the flattened index of the maximum value
            max_index = np.argmax(self.keypoints[i, :, :])

            # Convert the flattened index to row and column indices
            y, x = np.unravel_index(max_index, self.keypoints[i, :, :].shape)

            x_coords.append(x)
            y_coords.append(y)

        y_coords = np.array(y_coords)
        x_coords = np.array(x_coords)

        center_x, center_y = self.item['center']
        width = self.item['width']

        scale = self.item['scale'][0]
        
        # image shrunk by factor of 4, rediced by scale, and moved to bounding box. Must undo
        y_coords = 4*y_coords * scale + center_y - width // 2
        x_coords = 4*x_coords * scale + center_x - width // 2

        return np.array([x_coords, y_coords]).T      


    def merge_heatmaps(self):
        if self.keypoints is None or self.item is None:
            print("No heatmap available") 
        heatmap = np.zeros_like(self.keypoints[0,:,:])
        for i in range(10):
            heatmap = heatmap + self.keypoints[i, :, :]
        return heatmap


def test_model(model, path="./data/RealWorldBboxData/test_data.json"):
    import json, random



    with open(path, 'r') as f:
        data = dict(json.load(f))
    
    test_image = random.choice(list(data.keys()))
    img_data = data[test_image]
    
    img_dir = "data/RealWorldBboxData/color"
    img = cv2.imread(f"{img_dir}/{test_image}.png")

    bbox = img_data["bbox"]
    original = img.copy()

    t0 = time.time()
    predicted_keypoints = model.predict(img, bbox)
    print(f"Elapsed time: {time.time() - t0:.3f}, {test_image}")
    heatmap = model.merge_heatmaps()


    keypoints = img_data["keypoints"]

    predicted = original.copy()

    # Draw circles and add numbers
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    for i in range(10):
        num = str(i+1)
        text_size, _ = cv2.getTextSize(num, font, font_scale, font_thickness)

        kp = [int(keypoints[2*i]), int(keypoints[2*i+1])]
        cv2.circle(original, kp, 5, 0x0000FF, -1)
        
        text_pos_o = (kp[0] - text_size[0] // 2, kp[1] - 15)
        text_pos_p = (int(predicted_keypoints[i, 0]) -text_size[0]//2, int(predicted_keypoints[i, 1])-15)
        cv2.circle(predicted, (int(predicted_keypoints[i, 0]), int(predicted_keypoints[i, 1])), 5, 0x00FF00, -1)
        cv2.rectangle(predicted, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 0x00FFFF, 3)
        cv2.putText(original, num, text_pos_o, font, font_scale, 0x0000FF, font_thickness, cv2.LINE_AA)
        cv2.putText(predicted, num, text_pos_p, font, font_scale, 0x0000FF, font_thickness, cv2.LINE_AA)


    
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    heatmap = (255 * heatmap).astype(np.uint8)
    h, w = heatmap.shape
    heatmap = cv2.resize(heatmap, (4*w, 4*h))

    cv2.imshow("Original Keypoints", original)
    cv2.imshow("Predicted Keypoints", predicted)
    cv2.imshow("Keypoints heatmaps", heatmap)
    # cv2.imshow("Blurred image", blurr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import time
    model = keypoint_model()
    for i in range(10):
        test_model(model)
        
        # break
    
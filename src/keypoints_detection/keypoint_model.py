from __future__ import division
from __future__ import print_function




import torch
from torchvision import transforms
# from hourglass import hg
from PIL import Image
import os
#from os import join
from hourglass import hg
from time import time
import numpy as np
import cv2 

from transformations import *
from ..image_transformations.coordinate_transforms import IntrinsicsMatrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
Convert axis-angle representation to a 3x3 rotation matrix
"""
class Rodrigues(torch.autograd.Function):
    @staticmethod
    def forward(self, inp, device):
        pose = inp.detach().cpu().numpy()
        rotm, part_jacob = cv2.Rodrigues(pose)
        self.jacob = torch.Tensor(np.transpose(part_jacob)).contiguous().to(device)
        rotation_matrix = torch.Tensor(rotm.ravel()).to(device)
        return rotation_matrix.view(3, 3)

    @staticmethod
    def backward(self, grad_output):
        grad_output = grad_output.view(1,-1)
        grad_input = torch.mm(grad_output, self.jacob).to(device)
        grad_input = grad_input.view(-1)
        return grad_input

rodrigues = Rodrigues.apply


class keypoint_model:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform_list = [CropAndPad(out_size=(256, 256))]
        transform_list.append(ToTensor())
        transform_list.append(Normalize())

        self.transform = transforms.Compose(transform_list)

        self.model = torch.load('./models/keypoints_detection.pth')

        self.keypoints_2d = None
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
        confs = []
        for i in range(10):
            # Find the flattened index of the maximum value
            max_index = np.argmax(self.keypoints[i, :, :])

            # Convert the flattened index to row and column indices
            y, x = np.unravel_index(max_index, self.keypoints[i, :, :].shape)
            conf = np.max(self.keypoints[i,:,:])

            x_coords.append(x)
            y_coords.append(y)
            confs.append(conf)
        
        y_coords = np.array(y_coords)
        x_coords = np.array(x_coords)
        confs = np.array(confs)

        center_x, center_y = self.item['center']
        width = self.item['width']

        scale = self.item['scale'][0]
        
        # image shrunk by factor of 4, rediced by scale, and moved to bounding box. Must undo
        y_coords = 4*y_coords * scale + center_y - width // 2
        x_coords = 4*x_coords * scale + center_x - width // 2

        self.keypoints_2d = np.array([x_coords, y_coords, confs]).T 
        return self.keypoints_2d 


    def merge_heatmaps(self):
        if self.keypoints is None or self.item is None:
            print("No heatmap available") 
        heatmap = np.zeros_like(self.keypoints[0,:,:])
        for i in range(10):
            heatmap = heatmap + self.keypoints[i, :, :]
        return heatmap
    
    def predict_keypoints(self, K, keypoints3D):
        K = torch.from_numpy(K).to(self.device)
        keypoints3D = torch.from_numpy(keypoints3D).to(self.device)
        r = torch.rand(3, requires_grad=True, device=self.device) # rotation in axis-angle representation
        t = torch.rand(3 ,requires_grad=True, device=self.device)
        d = torch.from_numpy(self.keypoints_2d[:,2]).sqrt()[:, None].to(self.device)

        # print(d.shape)

        keypoints2d = torch.from_numpy(np.c_[self.keypoints_2d[:, :2], np.ones((self.keypoints_2d.shape[0], 1))].T).to(self.device)

        norm_keypoints_2d = torch.matmul(K.inverse(), keypoints2d).t()
        optimizer = torch.optim.Adam([r,t], lr=1e-2)

        converged = False
        rel_tol = 1e-7
        loss_old = 1e10
        while not converged:
            optimizer.zero_grad()
            R = rodrigues(r, self.device)
            k3d = torch.matmul(R, keypoints3D.transpose(1,0) + t[:, None])
            proj_keypoints = (k3d / k3d[2])[0:2,:].transpose(1,0)
            
            err = torch.norm(((norm_keypoints_2d[:, :2] - proj_keypoints) * d)**2, 'fro')
            err.backward()
            optimizer.step()
            if abs(err.detach() - loss_old) / loss_old < rel_tol:
                break
            else:
                loss_old = err.detach()

        R = rodrigues(r, self.device)
        return R[0].detach(), t.detach()




def test_model(model: keypoint_model, path="./data/RealWorldBboxData/test_data.json"):
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
    # K = np.array([
    #     [635.722,   0.,   630.956],
    #     [  0.,    635.722, 364.251],
    #     [  0.,      0.,      1.   ]
    # ])

    # S = np.load('kpt.npy')
    # pose = model.predict_keypoints(K, S)
    # print(pose)
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
        
        # text_pos_o = (kp[0] - text_size[0] // 2, kp[1] - 15)
        # text_pos_p = (int(predicted_keypoints[i, 0]) -text_size[0]//2, int(predicted_keypoints[i, 1])-15)
        cv2.circle(predicted, (int(predicted_keypoints[i, 0]), int(predicted_keypoints[i, 1])), 5, 0x00FF00, -1)
        cv2.rectangle(predicted, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), 0x00FFFF, 3)
        # cv2.putText(original, num, text_pos_o, font, font_scale, 0x0000FF, font_thickness, cv2.LINE_AA)
        # cv2.putText(predicted, num, text_pos_p, font, font_scale, 0x0000FF, font_thickness, cv2.LINE_AA)

    # 89mm


    dist_between(predicted_keypoints, keypoints)
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

def dist_between(predicted_keypoints, keypoints):
    print(f"Labeled Kaypoings:")
    for i in range(10):
        print(keypoints[2*i], keypoints[2*i+1])
    keypoints = np.array(predicted_keypoints).reshape(-1,2)
    for i in range(10):
        print(keypoints[i,:])
    print(f"predicted keypoints:")
    for i in range(10):
        print(predicted_keypoints[i, :2])

if __name__ == "__main__":
    import time
    model = keypoint_model()
    for i in range(1):
        test_model(model)
        
        # break
    
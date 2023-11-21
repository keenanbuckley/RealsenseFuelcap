from __future__ import division
from __future__ import print_function

import torch
from torchvision import transforms
# from hourglass import hg
from PIL import Image
import os
#from os import join
from keypoints_detection.hourglass import hg
from keypoints_detection.transformations import *

from time import time
import numpy as np
import cv2 
from torchvision.io import read_image
from scipy.spatial.transform import Rotation

from image_transformations.coordinate_transforms import calculate_matrix, IntrinsicsMatrix, annotate_img, TransformationMatrix
from bounding_box import BBoxModel



class KPModel:
    def __init__(self, path = './models/keypoints_detection.pth') -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform_list = [CropAndPad(out_size=(256, 256))]
        transform_list.append(ToTensor())
        transform_list.append(Normalize())

        self.transform = transforms.Compose(transform_list)

        #self.model = torch.load(path)
        self.model = hg(num_stacks=1, num_blocks=1, num_classes=10).to(self.device)
        checkpoint = torch.load('./models/model_checkpoint.pt')
        self.model.load_state_dict(checkpoint['model'])

        self.keypoints_2d = None
        self.keypoints = None
        self.item = None
        self.model.eval()

    def predict(self, image, bbox):
        '''
        Finds keypoints in image
        Args: 
            image: color image
            bbox: bounding box of fuel cap [xmin, ymin, xmax, ymax]
        Returns:
            2d array containing keypoints, with conficences for each point
        '''
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
        if not type(width) is np.float32:
            print(type(width))
            width = width.numpy()
        # image shrunk by factor of 4, reduced by scale, and moved to bounding box. Must undo
        y_coords = 4*y_coords * scale + center_y - width // 2
        x_coords = 4*x_coords * scale + center_x - width // 2

        self.keypoints_2d = np.array([x_coords, y_coords, confs]).T
        return self.keypoints_2d 


    def merge_heatmaps(self):
        """
        Creates heatmap, used for visualizations
        """
        if self.keypoints is None or self.item is None:
            print("No heatmap available") 
        heatmap = np.zeros_like(self.keypoints[0,:,:])
        for i in range(10):
            heatmap = heatmap + self.keypoints[i, :, :]
        return heatmap
    

    def predict_position(self, K : IntrinsicsMatrix, depth : np.ndarray, kernel_size:int=12, img=None, keypoints=None):
        """
        Calculates position and rotaiton of the fuel cap
        Args:
            K (IntrinsicsMatrix): Camera Intrinsics, used for pixel-point and point-pixel projections
            depth (np.ndarray): Depth image 
            kernel_size (int): size of the square used to get depth at each point
            img (optional): color image, will annotate image with keypoint locations and centerpoint
            keypoints (optional): can input keypoints manually but will default to class keypoints
        Returns:
            np.ndarray: rotation matrix
            np.ndarray: position vector
            np.ndattay: image with annotations, if none is provided, return none
        """
        
        if self.keypoints_2d is None and keypoints is None:
            print("No keypoints available")
            return None, None, None
        if self.keypoints_2d is None:
            kps = keypoints

        # take keypoints and calculate the position in 3D space using depth information
        kpts = self.keypoints_2d
        points3D = []
        for i in range(10):
            x,y = self.keypoints_2d[i, :2]
            xi,yi = [round(i) for i in [x,y]]
            depth_area = depth[yi-kernel_size//2:yi+kernel_size//2, xi-kernel_size//2:xi+kernel_size//2]
            max_depth = np.max(depth_area)
            ave_depth = np.mean(depth_area)

            if img is not None:
                cv2.rectangle(img, (xi-kernel_size//2, yi-kernel_size//2), (xi+kernel_size//2, yi+kernel_size//2), (0,255,255), 1)
            
            points3D.append(K.calc_position((x,y), max_depth))
        points3D = np.array(points3D)

        # calculate a plane using least squares approximation
        # z = ax+by+c
        A = np.c_[points3D[:, :2], np.ones_like(points3D[:, 0])]
        b = points3D[:, 2]
        x, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # 0 = ax+by-z+c
        # z axis = <a, b, -1> (normalized of course)
        z_axis = np.array([x[0], x[1], -1])
        z_axis /= np.linalg.norm(-z_axis)

        # calculate x axis using two point pairs that form lines paralel to the x axis
        x_axis = points3D[2, :] - points3D[0, :] + points3D[3, :] - points3D[1,:]
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_axis /= np.linalg.norm(x_axis)
        
        # y axis is z (cross) x
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        rotation = np.column_stack((x_axis, y_axis, z_axis))


        # calculate center point by connecting several points that intersect it and averaging their midpoints
        center_pts = [
            (points3D[0, :] + points3D[1, :]) / 2,
            (points3D[4, :] + points3D[8, :]) / 2,
            (points3D[5, :] + points3D[7, :]) / 2,
            (points3D[6, :] + points3D[9, :]) / 2

        ]
        ctr_pt = np.mean(center_pts, axis=0)
        if img is not None:
            ctr_px = K.calc_pixels(ctr_pt)
            cv2.circle(img, ctr_px, 5, (255,255,255), -1)
        
        return rotation, ctr_pt, img, residuals


def test_model(model: KPModel):
    import json, random, time
    import pandas as pd


    # img_dir = "./data/GroundTruth"
    # test_image = '1700088095696.png'#random.choice(os.listdir(img_dir))

    img_dir = "./data/RealWorldBboxData"
    
    test_image = random.choice(os.listdir(f"{img_dir}/color"))
    print(test_image)
    # img_data = data[test_image]


    img = read_image(f'{img_dir}/color/{test_image}')
    depth_img = np.load(f'{img_dir}/depth/{test_image[:-4]}.npy')
    K = IntrinsicsMatrix()

    t0 = time.time()
    bboxModel = BBoxModel("models/bbox_net_trained.pth")
    bbox, score = bboxModel.find_bbox(img)
    bbox = bbox.numpy()

    img = cv2.imread(f'{img_dir}/color/{test_image}')
    kpts = model.predict(img, bbox)
    rotation, translation, img, _ = model.predict_position(K, depth_img, kernel_size=12, img=img)
    if translation is not None:
        H = TransformationMatrix(R = rotation, t=translation)
        annotate_img(img, H, K)
    else:
        print("Could not detect center point")
    print("Elepsed time", time.time()-t0)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import time
    model = KPModel()
    for i in range(5):
        test_model(model)
        
        # break
    
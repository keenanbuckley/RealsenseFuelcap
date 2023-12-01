################################################
######### FILE IS NOT COMPLETE #################
################################################
# TODO: VERYIFY intrinsics matrix
# TODO: Verity translation matrix

from typing import List, Union

import json
import numpy as np
from scipy.spatial.transform import Rotation
import cv2

with open('./data/camera_intrinsics.json', 'r') as f:
    camera_instrinsics = dict(json.load(f))

class TransformationMatrix:
    """
    Transformation Matrix Class
    """
    def __init__(self, R=np.eye(3), t = np.zeros(3), H: np.ndarray=None) -> None:
        if H is None:
            self.matrix = TransformationMatrix.__transformation_matrix(t, R)
        else:
            assert H.shape == (4,4)
            self.matrix = H

    def set_rotation(self, R=np.eye(3)) -> None:
        assert R.shape == (3,3)
        self.matrix = TransformationMatrix.__transformation_matrix(self.matrix[:3, 3], R)

    def set_translation(self, t=np.zeros(3)) -> None:
        assert len(t) == 3
        self.matrix = TransformationMatrix.__transformation_matrix(t, self.matrix[:3,:3])

    def transform_point(self, pose=np.zeros(3)) -> np.ndarray:
        assert len(pose) == 3 or len(pose) == 4
        
        if len(pose) == 3:
            pose = np.append(pose, [1])
        else:
            assert pose[3] == 1
        
        return (self.matrix @ pose)[:3]

    def inverse_transform(self, pose=np.zeros(3)) -> 'TransformationMatrix':
        mat_copy = self.matrix.copy()

        self.matrix = np.linalg.inv(self.matrix)
        pose = self.transform(pose)

        self.matrix = mat_copy

        return pose

    def as_mat(self):
        return self.matrix
    
    def as_pos_and_quat(self):
        R = self.matrix[:3,:3]
        orientation = Rotation.from_matrix(R).as_quat()
        pose = self.matrix[:3,3]
        return pose, orientation

    def invert(self):
        return TransformationMatrix( H = np.linalg.inv(self.matrix) )


    def __mul__(self, H) -> 'TransformationMatrix':
        
        if isinstance(H, np.ndarray):

            if H.shape == (4,4):
                return TransformationMatrix(H=self.matrix @ H)
            
            elif H.shape == (3,3):
                H = TransformationMatrix(R=H)
                return TransformationMatrix(H=self.matrix @ H.matrix)
            
            elif H.shape == (3,) or H.shape == (3,1) or H.shape == (1,3):
                
                H = TransformationMatrix(t = H)
                return TransformationMatrix(H=self.matrix @ H.matrix)
            
        elif isinstance(H, TransformationMatrix):

            if H.matrix.shape == (4,4):
                return TransformationMatrix(H=self.matrix @ H.matrix)
            
        raise ValueError("Must multiply by a 4x4 numpy array of Transformation Matrix")

    @ staticmethod
    def __transformation_matrix(translation : np.ndarray = np.zeros(3), rotation : np.ndarray = np.eye(3)) -> np.ndarray:
        """
        assembles transformation matrix from rotation and translation

        Args:
            translation (np.ndarray): 3x1 translation matrix
            rotation (np.ndarray): 3x3 rotation matrix
        Returns:
            np.ndarray: 4x4 translation matrix
        """
        # ensure rotation matrix is orthonormal
        #assert np.allclose(np.dot(rotation, rotation.T), np.eye(3))

        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] =  translation# translation
        matrix[:3, :3] = rotation
        return matrix


class IntrinsicsMatrix:
    '''
    Class defining intrinsics matrix
    contains 2 methods
    calc_pixels: calculate pixel values from positions
    calc_position: calculates positon from pixel coordinates
    '''

    def __init__(self, fov_x = 84, dimensions = (720,1280), degrees=True) -> None:
        '''
        Contstructor to create intrinsics matrix
        Default parameters mirror the d405 camera with 720p resolution
        args:
            fov_x: field of view in the horizintal direction of the camera
            dimensions (tuple): pair o heigh and width of image 
        '''
        assert len(dimensions) == 2
        assert fov_x > 0

        h, w = dimensions
        cy, cx = h / 2, w / 2
        
        if degrees:
            fov_x = np.deg2rad(fov_x)
        
        fl = (w/2) / np.tan(fov_x / 2)

        fx = float(camera_instrinsics['rectified.1.fx'])
        fy = float(camera_instrinsics['rectified.1.fy'])
        cx = float(camera_instrinsics['rectified.1.ppx'])
        cy = float(camera_instrinsics['rectified.1.ppy'])

        self.matrix = np.array([
            [fx,    0,      cx],
            [0,     fy,     cy],
            [0,     0,      1 ],
        ], dtype=np.float32)

        assert self.matrix.shape == (3,3)

    def calc_pixels(self, pose=[0,0,1]):
        if type(pose) is list:

            pose = np.array(pose, dtype=np.float32)
        # print(pose)
        if pose.shape[0] != 3:
            raise ValueError("Position must be in 3D cartesian Coordinates -> ", pose)
        if pose[2] <= 0:
            raise ValueError("Z coordinate must be positive to be seen by the camera, z =", pose[2])

        coords = self.matrix @ pose

        pixels = (coords / coords[2])[:2]
        return [round(p) if not np.isnan(p) else 0 for p in pixels]
    
    def calc_position(self, pixels: tuple = (640, 360), depth=1):
        '''
        Calculates position based on position and depth
        If no pixels give, assume default images center
        If no depth is give, assume the depth is 1 (unit)
        Ards:
            pixels (tuple): (x,y) coordinate on the image
            depth: depth of the point from the camera
        Returns:
            np.array: (3x3) position (x,y,z)
        '''
        if len(pixels) != 2:
            raise ValueError(f"Must give pixels in 2D cartesian coordinates: ({pixels[0]},{pixels[1]})")
        if depth <= 0:
            raise ValueError("Depth must be a positive number to be seen by camera ->", depth)

        pixels = depth * np.array(list(pixels) + [1])
        return np.linalg.inv(self.matrix) @ pixels


def calculate_matrix(x: float, y: float, z: float, angle_mount: float = 0, angle_cap: float = 0, units:str="in") -> TransformationMatrix:
    """
    Calculates the transformation matrix from the camera to the fuel cap using the data collection rig
    Assumes fuel cap coordinates to be positive x in the left direction on wall, positive y down, positive z towards camera
    Positive fuel cap rotation are counter-clockwise, positve camera mount rotations are counter-clockwise (based on coordinates)
    
    Args:
        x (float): horizontal distance along the wall
        y (float): virtical distance along the wall
        z (float): distance from the rig to the wall
        angle_mount (float): angle measured at the camera mount
        angle_cap (float): angle of the fuel cap on the wall (use protractor)
        inits (in): the unit system 
    Returns:
        np.ndarray: translation matrix from camera to fuel cap
    """

    if units not in ["in", "mm", "cm"]:
        raise ValueError("Unit system must be in inches, centimeters, or milimeters ->", units)
    
    if units == "in":
        x,y,z = [i * 25.4 for i in [x,y,z]]
    if units == "cm":
        x,y,z = [i * 10 for i in [x,y,z]]

    # shrink y by height of the mount, which is 1.25 inches
    y -= 25.4 * (1.2)

    # world2left transformation matrix
    world2left = np.array(
        [[float(camera_instrinsics['world2left_rot.x.x']), float(camera_instrinsics['world2left_rot.x.y']), float(camera_instrinsics['world2left_rot.x.z'])],
        [float(camera_instrinsics['world2left_rot.y.x']), float(camera_instrinsics['world2left_rot.y.y']), float(camera_instrinsics['world2left_rot.y.z'])],
        [float(camera_instrinsics['world2left_rot.z.x']), float(camera_instrinsics['world2left_rot.z.y']), float(camera_instrinsics['world2left_rot.z.z'])],]
    )
    
    H = TransformationMatrix()
    H = H * Rotation.from_euler('z', -angle_cap, degrees=True).as_matrix()
    #H = H * Rotation.from_euler('y', -10, degrees=True).as_matrix()
    H = H * np.array([x,y,z])
    H = H * Rotation.from_euler('y', angles=180-angle_mount, degrees=True).as_matrix()
    H = H * __mount_to_camera_translation()
    H = H * world2left


    return H.invert()    


def annotate_img(img: np.ndarray, H: TransformationMatrix, K: IntrinsicsMatrix, line_width=3, axis_len = 30) -> np.ndarray:
    """
    Plots coordinate axis on image

    Args:
        img (np.ndarray): image array
        translation (np.ndarray): 4x4 translation matrix
        K (np.ndarray): camera intrinsic matrix

    Returns:
        np.ndarray: annotated image
    """
    # initialize points for coordinate axis
    points = [
        np.array([0, 0, 0, 1]),
        np.array([axis_len, 0, 0, 1]),
        np.array([0, axis_len, 0, 1]),
        np.array([0, 0, axis_len, 1])
    ]

    pixels = [K.calc_pixels(H.transform_point(p)) for p in points]

    o,x,y,z = pixels

    thickness = line_width
    cv2.line(img, o, x, (255,255,255), thickness+3)
    cv2.line(img, o, x, (0,0,255), thickness)
    cv2.line(img, o, y, (255,255,255), thickness+3)
    cv2.line(img, o, y, (0,255,0), thickness)
    cv2.line(img, o, z, (255,255,255), thickness+3)
    cv2.line(img, o, z, (255,150,0), thickness)

    return img


def __mount_to_camera_translation(units: str="mm") -> np.ndarray:
    """
    camera translation based on datasheet information
    camera origin is located on left camera, inset 3.07 mm
    Default to return translation in centimeters
    """
    __check_inits(units)
    # return np.zeros(3)
    x = float(camera_instrinsics['baseline'])/2     # cameras are 18 mm apart
    y = -42 / 2                                     # cameras are located on middle of camera in y, and cam is 42 mm tall
    z = 23 - 8.35 - 3.7                             # z distance between mounting hole and z = 0 on depth module

    trans = np.array([x,y,z], dtype = np.float32)
    
    return trans

def __check_inits(units):
    if units not in ["mm", "cm", "in"]:
        raise ValueError("Units must be centimeters, milimeeters or inches ->", units)


def main():
    K = IntrinsicsMatrix()
    pix = [i for i in K.matrix[:2, 2]]
    
    depth = 500
    new_pos = K.calc_position((pix[0]+10, pix[1]), depth)
    new_pixels = K.calc_pixels(new_pos)
    print(new_pixels, pix)
    # pos = [1, 1, 2]
    # pixels = (0,0)
    # depth = 3
    # target_pos = [-3.33183754, -1.87415862,  3.]
    # tol = 1e-5


    # # tests
    # assert K.calc_pixels() == [640, 360]
    # assert K.calc_pixels(pos) == [928, 648]
    # assert [int(i) for i in list(K.calc_position())] == [0, 0, 1]
    # pos = K.calc_position(pixels, depth)
    # assert np.sum(pos - np.array(target_pos)) < tol


    # img = cv2.imread("data/saved_img.png")
    # K = IntrinsicsMatrix()
    # translation = calculate_matrix(-6, 19.05, 45.72, angle_mount=-10, angle_cap=20)
    # # pos, orien = translation.as_pos_and_quat()

    # # print(pos, orien)

    # rot = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()

    # img = annotate_img(img, translation, K, axis_len=30)

    # cv2.imshow("Annotated Image", img)
    # cv2.waitKey(0)

    # cv2.imwrite("annotated_img.png", img)


if __name__ == "__main__":
    main()
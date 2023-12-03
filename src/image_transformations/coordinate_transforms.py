##########################################################
############### Calculates 2D-3D and 3D-3D ###############
##########################################################

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
    def __init__(self, R:np.ndarray=np.eye(3), t:np.ndarray = np.zeros(3), H: np.ndarray=None) -> None:
        """Constructor for transformation matrix

        Args:
            R (np.ndarray, optional): Rotation matrix. Defaults to np.eye(3) (no rotation).
            t (np.ndarray, optional): Translation vector. Defaults to np.zeros(3) (no translation).
            H (np.ndarray, optional): Transformation matrix. Defaults to None.

        """
        # if H is defined, use that, otherwise, set to translation and rotation
        if H is None:
            self.matrix = TransformationMatrix.__transformation_matrix(t, R)
        else:
            if H.shape != (4,4):
                raise ValueError("Transformation matrix must be 4x4")
        
            self.matrix = H

    def set_rotation(self, R=np.eye(3)) -> None:
        if R.shape != (3,3):
            raise ValueError("Rotation must be a 3x3 matrix")
        
        self.matrix = TransformationMatrix.__transformation_matrix(self.matrix[:3, 3], R)

    def set_translation(self, t=np.zeros(3)) -> None:
        if len(t) != 3:
            raise ValueError("Translation must have 3 dimensions")
        self.matrix = TransformationMatrix.__transformation_matrix(t, self.matrix[:3,:3])

    def transform_point(self, position:np.ndarray=np.zeros(3)) -> np.ndarray:
        """Transforms point based on position

        Args:
            position (np.ndarray, optional): position in original reference frame. Defaults to np.zeros(3).

        Raises:
            ValueError: position must be 3 or 4 values long

        Returns:
            np.ndarray: transformed position in new reference frame
        """
        if not (len(position) == 3 or len(position) == 4):
               raise ValueError("Pose must be either (x,y,z) vector or normalized (x,y,z,1) vector")
        
        if len(position) == 3:
            position = np.append(position, [1])
        else:
            assert position[3] == 1
        
        return (self.matrix @ position)[:3]

    def inverse_transform(self, pose:np.ndarray=np.zeros(3)) -> 'TransformationMatrix':
        """Calculates inverse transform 

        Args:
            pose (np.ndarray, optional): 3x1 pose or 4x1 normalized pose (x,y,z,1). Defaults to np.zeros(3).

        Returns:
            TransformationMatrix: _description_
        """
        mat_copy = self.matrix.copy()

        self.matrix = np.linalg.inv(self.matrix)
        pose = self.transform(pose)

        self.matrix = mat_copy

        return pose

    def as_mat(self) -> np.ndarray:
        """Return Just the matrix

        Returns:
            np.ndarray: 4x4 transformation matirix as matrix
        """
        return self.matrix
    
    def as_pos_and_quat(self):
        """_summary_

        Returns:
            tuple(np.ndarray): pose and orientation as a tuple of np vectors ( position, quaternion )
        """
        R = self.matrix[:3,:3]
        orientation = Rotation.from_matrix(R).as_quat()
        pose = self.matrix[:3,3]
        return pose, orientation

    def invert(self) -> 'TransformationMatrix':
        """Return an inverted transformation matrix

        Returns:
            TransformationMatrix: inverted matrix
        """
        return TransformationMatrix( H = np.linalg.inv(self.matrix) )


    def __mul__(self, H) -> 'TransformationMatrix':
        """Defines multiplication operator. Allows for cleaner multiplication opperations

        Args:
            H (TransformationMatrix or np.ndarray): Transformation Matrix, Rotation Matrix, or translation vector to transform by

        Raises:
            ValueError: Can only multiply by 4x4 transformation matrix, 3x3 rotation matrix, or 3x1/1x3 translation matrix

        Returns:
            TransformationMatrix: Result of matrix multiplication
        """
        
        # Defines how to handle numpy arrays
        if isinstance(H, np.ndarray):
            
            # if a 4x4 matrix is passed, assume H is uninstantiated transformation matrix
            if H.shape == (4,4):
                
                return TransformationMatrix(H=self.matrix @ H)
            
            # if a 3x3 matrix is passed, assume H is rotation matrix
            elif H.shape == (3,3):

                H = TransformationMatrix(R=H)    # initialize transformaton matrix with rotation
                return TransformationMatrix(H=self.matrix @ H.matrix)
            
            # if a 3x1 or 1x3 matrix is passed, assume H is translation vector
            elif H.shape == (3,) or H.shape == (3,1) or H.shape == (1,3):

                H = TransformationMatrix(t = H)     # initialize tranformation matrix with single translation
                return TransformationMatrix(H=self.matrix @ H.matrix)

        # defines how to handle TransformationMatrix multiplication
        elif isinstance(H, TransformationMatrix):
            # multiply matricies
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
    """
    Class defining intrinsics matrix
    contains 2 methods
    calc_pixels: calculate pixel values from positions
    calc_position: calculates positon from pixel coordinates
    """

    def __init__(self, dimensions = (720,1280)) -> None:
        """
        Contstructor to create intrinsics matrix
        Default parameters mirror the d405 camera with 720p resolution
        Intrinsics information is based on a previous calibration from camera

        Args:
            dimensions (tuple, optional): _description_. Defaults to (720,1280).
        Raises:
            ValueError: dimensions need height and width
        """
        
        if len(dimensions) != 2:
            raise ValueError("Camera dimensions must be 2D, (height, width)")

        h, w = dimensions
        cy, cx = h / 2, w / 2

        fx = float(camera_instrinsics['rectified.1.fx'])
        fy = float(camera_instrinsics['rectified.1.fy'])
        cx = float(camera_instrinsics['rectified.1.ppx'])
        cy = float(camera_instrinsics['rectified.1.ppy'])

        self.matrix = np.array([
            [fx,    0,      cx],
            [0,     fy,     cy],
            [0,     0,      1 ],
        ], dtype=np.float32)

        if self.matrix.shape != (3,3):
            raise ValueError("Intrinsics matrix must be a 3x3 matrix, something went wrong")

    def calc_pixels(self, position=[0,0,1]):
        """Calculates pixel coordinates from 3D cartesian coordinates

        Args:
            position (list, optional): 3D position. Defaults to [0,0,1].

        Raises:
            ValueError: position must have 3 values (x,y,z)
            ValueError: Z must be positive position[2] > 0

        Returns:
            list[list[int]]: Pixel coordinates in [ [u1,v1],[u2,v2],...,[un,vn] ]
        """

        # convert position to numpy array if given as list
        if type(position) is list:
            position = np.array(position, dtype=np.float32)

        if position.shape[0] != 3:
            raise ValueError("Position must be in 3D cartesian Coordinates -> ", position)
        if position[2] <= 0:
            raise ValueError("Z coordinate must be positive to be seen by the camera, z =", position[2])
        
        # calculate pixel position, normalize
        coords = self.matrix @ position
        pixels = (coords / coords[2])[:2]

        # return pixel position, if any are nan set to 0
        return [round(p) if not np.isnan(p) else 0 for p in pixels]
    
    def calc_position(self, pixels: tuple = (640, 360), depth=1):
        """calculates position in frame based on depth

        Args:
            pixels (tuple, optional): (x,y) coordinate on the image. Defaults to (640, 360).
            depth (int, optional): depth of the point from the camera. Defaults to 1.

        Raises:
            ValueError: Must give x,y coordinates of pixels
            ValueError: Depth must be positive

        Returns:
            np.array: (3x3) position (x,y,z)
        """

        if len(pixels) != 2:
            raise ValueError(f"Must give pixels in 2D cartesian coordinates: ({pixels[0]},{pixels[1]})")
        if depth <= 0:
            raise ValueError("Depth must be a positive number to be seen by camera ->", depth)

        # calculate 3D coordinates for pixels using the invers of intrinsics matrix and multiply by depth
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
        units (in): the unit system 
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

    # world2left rotation matrix, obtained from camera intrinsics
    world2left = np.array(
        [[float(camera_instrinsics['world2left_rot.x.x']), float(camera_instrinsics['world2left_rot.x.y']), float(camera_instrinsics['world2left_rot.x.z'])],
        [float(camera_instrinsics['world2left_rot.y.x']), float(camera_instrinsics['world2left_rot.y.y']), float(camera_instrinsics['world2left_rot.y.z'])],
        [float(camera_instrinsics['world2left_rot.z.x']), float(camera_instrinsics['world2left_rot.z.y']), float(camera_instrinsics['world2left_rot.z.z'])],]
    )
    
    H = TransformationMatrix()
    H = H * Rotation.from_euler('z', -angle_cap, degrees=True).as_matrix()
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
        np.array([0, 0, 0, 1]),             # origin
        np.array([axis_len, 0, 0, 1]),      # x axis
        np.array([0, axis_len, 0, 1]),      # y axis
        np.array([0, 0, axis_len, 1])       # z axis
    ]

    # calculate pixel coordinates by first transforming point by H, then projecting them onto image with K
    pixels = [K.calc_pixels(H.transform_point(p)) for p in points]

    o,x,y,z = pixels

    thickness = line_width

    # draw lines connecting origin to each axis. Add a white line to contrast dark backgrounds (not necessary if unwanted)
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
################################################
######### FILE IS NOT COMPLETE #################
################################################
# TODO: VERYIFY intrinsics matrix
# TODO: Verity translation matrix

from typing import List, Union

import numpy as np
from scipy.spatial.transform import Rotation
import cv2


def calculate_matrix(x: float, y: float, z: float, angle_mount: float = 0, angle_cap: float = 0) -> np.ndarray:
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
    
    Returns:
        np.ndarray: translation matrix from camera to fuel cap
    """
    R_cap = Rotation.from_euler('z',angle_cap, degrees=True).as_matrix()
    H_cap = __transformation_matrix(rotation=R_cap)

    t_cap_to_mount = np.array([x,y,z])
    R_cap_to_mount = Rotation.from_euler('y',angle_mount+180, degrees=True).as_matrix()
    H_cap_to_mount = __transformation_matrix(t_cap_to_mount, R_cap_to_mount)

    t_mount_to_cam = __mount_to_camera_translation()
    H_mount_to_cam = __transformation_matrix(translation=t_mount_to_cam)

    H_cap_to_cam = H_cap @ H_cap_to_mount @ H_mount_to_cam
    return np.linalg.inv(H_cap_to_cam)

    # t_mount_to_cap = np.array([x, y, z])
    # R_mount_to_cap = Rotation.from_euler('z',angle_cap,degrees=True).as_matrix()
    # H_mount_to_cap = __transformation_matrix(t_mount_to_cap, R_mount_to_cap)                          # rotation
    
    
    # # t_camera_to_mount = np.zeros(3)
    # t_camera_to_mount = __mount_to_camera_translation()
    # R_camera_to_mount = Rotation.from_euler('y',180+angle_mount,degrees=True).as_matrix()
    # H_camera_to_mount = __transformation_matrix(t_camera_to_mount, R_camera_to_mount)

    # H_cap_to_cam = H_mount_to_cap @ H_camera_to_mount
    # H_cam_to_cap = np.linalg.inv(H_cap_to_cam)
    # return H_cam_to_cap


def matrix_to_pos(transformation: np.ndarray) -> List[Union[np.ndarray, np.ndarray]]:
    '''
    Convert a 4x4 transformation matrix into position and orientation (quaternion)

    Args:
        transformation (np.ndarray): 4x4 transformation matrix

    Returns:
        np.ndarray: position (cartesian)
        np.ndarray: orientation (quaternion)
    
    '''
    position = transformation[:3, 3]
    rotation = Rotation.from_matrix(transformation[:3, :3])
    quaternion = rotation.as_quat()

    return [position, quaternion]


def pos_to_matrix(position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(orientation)
    return __transformation_matrix(position, rotation)


def annotate_img(img: np.ndarray, translation: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Plots coordinate axis on image

    Args:
        img (np.ndarray): image array
        translation (np.ndarray): 4x4 translation matrix
        K (np.ndarray): camera intrinsic matrix

    Returns:
        np.ndarray: annotated image
    """
    axis_len = 5
    
    # initialize points for coordinate axis
    origin = np.array([0, 0, 0, 1])
    x_coord = np.array([axis_len, 0, 0, 1])
    y_coord = np.array([0, axis_len, 0, 1])
    z_coord = np.array([0, 0, axis_len, 1])

    # calculate points in 3D space
    origin = np.matmul(translation, origin)
    x_coord = np.matmul(translation, x_coord)
    y_coord = np.matmul(translation, y_coord)
    z_coord = np.matmul(translation, z_coord)

    # calculate points from camera intrinsics
    orig = np.matmul(K, origin[:3])
    x = np.matmul(K, x_coord[:3])
    y = np.matmul(K, y_coord[:3])
    z = np.matmul(K, z_coord[:3])

    #normalize points
    orig = np.array(np.round(orig / orig[2])[:2], dtype=np.int32)
    x = np.array(np.round(x / x[2])[:2], dtype=np.int32)
    y = np.array(np.round(y / y[2])[:2], dtype = np.int32)
    z = np.array(np.round(z / z[2])[:2], dtype=np.int32)


    thickness = 2
    cv2.line(img, orig, x, (0,0,255), thickness)
    cv2.line(img, orig, y, (0,255,0), thickness)
    cv2.line(img, orig, z, (255,0,0), thickness)


    return img


def __mount_to_camera_translation(cm=True) -> np.ndarray:
    """
    camera translation based on datasheet information
    camera origin is located on left camera, inset 3.07 mm
    Default to return translation in centimeters
    """
    x = -9          # cameras are 18 mm apart
    y = 42 / 2          # cameras are located on middle of camera in y, and cam is 42 mm tall
    z = -3.7

    trans = np.array(np.array([x,y,z]), dtype = np.float32)
    if cm:
        return trans / 10
    return trans


def __transformation_matrix(translation : np.ndarray = np.zeros(3), rotation : np.ndarray = np.eye(3)) -> np.ndarray:
    """
    assembles transformation matrix from rotation and translation

    Args:
        translation (np.ndarray): 3x1 translation matrix
        rotation (np.ndarray): 3x3 rotation matrix
    Returns:
        np.ndarray: 4x4 translation matrix
    """
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] =  translation# translation
    matrix[:3, :3] = rotation
    return matrix


def __intrinsics_matrix(dimensions : np.ndarray = np.array([1280, 720]), focal_length_mm : float = 1.93, fov : float = 84) -> np.ndarray:
    """
    Defines the camera intrinsics matrics. Based off pinhole model, assumes focal length is the same in x and y. Assumes dimensions of 720x1280 unless specified.
     Assumes focal length of 1.93 mm unless specified.
     Assumes fov 84 degrees unless specified.

    Args: 
        dimensions (np.array): dimensions of the image
        focal_length_mm (float): focal length of camera in mm
        fov (float): horizontal field of view
    Returns:
        np.ndarray: 3x3 intrinsics matrix
    """
    h, w = dimensions
    focal_length_in_pixels = (w / 2) * np.tan( np.deg2rad(fov / 2) )

    print(f"width: {w}, height: {h}")
    return np.array([
        [focal_length_in_pixels, 0, w/2],
        [0, focal_length_in_pixels, h/2],
        [0,0,1]
    ])




def main():
    # position, quaternion = matrix_to_pos(np.array([
    #     [1, 0, 0, 1],
    #     [0, 1, 0, 2],
    #     [0, 0, 1, 3],
    #     [0, 0, 0, 1],
    # ]))
    # # print(position)
    # # print(quaternion)
    # assert np.array_equal(position, np.array([1, 2, 3]))
    # assert np.array_equal(quaternion, np.array([0, 0, 0, 1]))

    # rotation = np.eye(3)
    # position = np.transpose(np.ones(3))
    # H = __transformation_matrix(position, rotation)
    # assert np.array_equal(H, np.array([
    #     [1, 0, 0, 1],
    #     [0, 1, 0, 1],
    #     [0, 0, 1, 1],
    #     [0, 0, 0, 1],
    # ]))

    # print(calculate_matrix(20, 30, -30, angle_mount=20))

    # img = np.ones((720, 1280, 3)) * 255
    img = cv2.imread("/home/mines/mines_ws/data/image.png")
    
    K = __intrinsics_matrix(dimensions=img.shape[:2])

    translation = calculate_matrix(-6, 19.05, 45.72, angle_mount=-10, angle_cap=20)
    # print(np.matmul(translation, np.array([0,0,0,1])))


    pos, orien = matrix_to_pos(translation)
    # print(pos)
    # print(orien)
    img = annotate_img(img, translation, K)

    cv2.imshow("Annotated Image", img)
    cv2.waitKey(0)

    cv2.imwrite("annotated_img.png", img)


if __name__ == "__main__":
    main()
################################################
######### FILE IS NOT COMPLETE #################
################################################


from typing import List, Union

import numpy as np
from scipy.spatial.transform import Rotation
import pytest


def camera_translation() -> np.ndarray:
    # camera translation based on datasheet information
    # camera origin is located on left camera, inset 3.07 mm
    x = 18 / 2          # cameras are 18 mm apart
    y = 42 / 2          # cameras are located on middle of camera in y, and cam is 42 mm tall
    z = -3.07
    return np.transpose(np.array([x,y,z]))

def calculate_matrix(x: float, y: float, z: float, angle_mount: float = 0, angle_cap: float = 0) -> np.ndarray:
    t_mount_to_cap = np.transpose(np.array([x, y, z]))
    R_mount_to_cap = Rotation.from_euler('z',angle_cap,degrees=True).as_matrix()
    H_mount_to_cap = transformation_matrix(t_mount_to_cap, R_mount_to_cap)                          # rotation
    

    t_camera_to_mount = np.transpose(np.zeros(3))
    R_camera_to_mount = Rotation.from_euler('y',angle_mount,degrees=True).as_matrix()
    H_camera_to_mount = transformation_matrix(t_camera_to_mount, R_camera_to_mount)

    H_cap_to_cam = np.matmul(H_mount_to_cap, H_camera_to_mount)
    H_cam_to_cap = np.linalg.inv(H_cap_to_cam)
    return H_cam_to_cap


def transformation_matrix(translation : np.ndarray, rotation : np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] =  translation# translation
    matrix[:3, :3] = rotation
    return matrix



def matrix_to_pos(transformation: np.ndarray) -> List[Union[np.ndarray, np.ndarray]]:
    '''Convert a 4x4 transformation matrix into position (cartesian) and orientation (quaternion)'''
    position = transformation[:3, 3]
    rotation = Rotation.from_matrix(transformation[:3, :3])
    quaternion = rotation.as_quat()

    return [position, quaternion]


def main():
    position, quaternion = matrix_to_pos(np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 2],
        [0, 0, 1, 3],
        [0, 0, 0, 1],
    ]))
    # print(position)
    # print(quaternion)
    assert np.array_equal(position, np.array([1, 2, 3]))
    assert np.array_equal(quaternion, np.array([0, 0, 0, 1]))

    rotation = np.eye(3)
    position = np.transpose(np.ones(3))
    H = transformation_matrix(position, rotation)
    assert np.array_equal(H, np.array([
        [1, 0, 0, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
    ]))

    print(calculate_matrix(20, 30, -30, angle_mount=20))



if __name__ == "__main__":
    main()
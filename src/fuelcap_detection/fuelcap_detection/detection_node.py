import cv2
import numpy as np
import rclpy
import time
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from custom_interfaces.srv import CaptureImage

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from rclpy.parameter import Parameter, ParameterMsg, ParameterType, ParameterValue

import sys
import os

class DetectionNode(Node):
    def __init__(self, exposure: int = 7500, annotations=False):
        super().__init__('fuelcap_detection')
        self.bridge = CvBridge()

        # Parameters to automatically change for the realsense node
        camera_parameters = list()
        camera_parameters.append(ParameterMsg(name='depth_module.exposure', value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=exposure)))
        
        # Wait until realsense node is active, then change its parameters
        msg = SetParameters.Request()
        msg.parameters = camera_parameters
        self.cli = self.create_client(SetParameters, '/camera/camera/set_parameters')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('realsense service not available, waiting again...')
        self.cli.call_async(msg)

        # Create subscribers to the realsense color and depth topics
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/color/image_rect_raw',
            self.color_listener_callback,
            10)
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.depth_listener_callback,
            10)
        
        # For storing image messages while waiting for the other to arrive
        self.color_img_msg = None
        self.depth_img_msg = None

        # Initialize the bounding box and keypoints DL models
    
    def color_listener_callback(self, img_msg: Image):
        self.color_img_msg = img_msg
        if not self.depth_img_msg is None:
            self.rgbd_callback()
            self.color_img_msg = None
            self.depth_img_msg = None
    
    def depth_listener_callback(self, img_msg: Image):
        self.depth_img_msg = img_msg
        if not self.color_img_msg is None:
            self.rgbd_callback()
            self.color_img_msg = None
            self.depth_img_msg = None
    
    def rgbd_callback(self):
        try:
            image_color = self.bridge.imgmsg_to_cv2(self.color_img_msg, desired_encoding="bgr8")
            image_depth = self.bridge.imgmsg_to_cv2(self.depth_img_msg, desired_encoding="16UC1")
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

    def publish_annotated_image(self, annotated_image):
        pass
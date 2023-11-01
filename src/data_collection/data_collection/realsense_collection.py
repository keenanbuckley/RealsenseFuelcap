import cv2
import numpy as np
import rclpy
import time
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from rclpy.parameter import Parameter

import sys


class Realsense_Simulated_Collection(Node):
    def __init__(self, n_pictures : int):
        super().__init__('realsense_subscriber')
        self.bridge = CvBridge()
        self.n_pictures = n_pictures

        parameters = list()
        parameters.append(Parameter('depth_module.enable_auto_exposure', Parameter.Type.BOOL, True))
        parameters.append(Parameter('depth_module.auto_exposure_roi.top', Parameter.Type.INTEGER, 65))
        parameters.append(Parameter('depth_module.auto_exposure_roi.bottom', Parameter.Type.INTEGER, 720-65))
        parameters.append(Parameter('depth_module.auto_exposure_roi.left', Parameter.Type.INTEGER, 75))
        parameters.append(Parameter('depth_module.auto_exposure_roi.right', Parameter.Type.INTEGER, 1280-75))

        msg = SetParameters.Request()
        print(msg.get_fields_and_field_types())
        #msg.parameters = parameters

        #self.cli = self.create_client(SetParameters, '/camera/camera/set_parameters')
        #while not self.cli.wait_for_service(timeout_sec=1.0):
        #    self.get_logger().info('service not available, waiting again...')
        #self.cli.call_async(msg)

        # Update the topic names based on your simulated camera's topics
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

        self.color_img_msg = None
        self.depth_img_msg = None

    def keyboard_listener(self):
        timestamp = int(1000*time.time())

        try:
            image_color = self.bridge.imgmsg_to_cv2(self.color_img_msg, desired_encoding="bgr8")
            image_depth = self.bridge.imgmsg_to_cv2(self.depth_img_msg, desired_encoding="16UC1")
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        # Convert depth image to np array
        image_depth = np.array(image_depth, dtype=np.uint16)

        # Save color image
        cv2.imwrite(f'data/color/{timestamp}.png', image_color)

        # Save depth image
        np.save(f'data/depth/{timestamp}.npy', image_depth)

        # Log the images in image_list.txt
        with open(f'image_list.txt', 'a') as f:
            f.write(f'{timestamp}\n')

        # Reset images to None
        self.color_img_msg = None
        self.depth_img_msg = None

        # Print depth info
        #self.get_logger().info(f"Depth Max: {np.max(image_depth)}, Depth Min: {np.min(image_depth)}")
        #self.get_logger().info(f"Depth Shape: {image_depth.shape[0]}x{image_depth.shape[1]}")
        #self.get_logger().info(f"Depth at center: {image_depth[int(image_depth.shape[0]/2), int(image_depth.shape[1]/2)]}")

    def color_listener_callback(self, img_msg):
        self.color_img_msg = img_msg
        if not self.depth_img_msg is None:
            self.keyboard_listener()
        #self.get_logger().info(f'Image of shape: {self.color_img_msg.width}x{self.color_img_msg.height}')
    
    def depth_listener_callback(self, img_msg):
        self.depth_img_msg = img_msg
        if not self.color_img_msg is None:
            self.keyboard_listener()


def main(args=None):
    py_args = sys.argv[1:]
    try:
        n = int(py_args[0])
        print(f"Collecting {n} pictures")
    except:
        print("No arguments detected, collecting 1000 pictures")
        n = 1000
    rclpy.init(args=args)


    realsense_subscriber = Realsense_Simulated_Collection(n_pictures=n)

    rclpy.spin(realsense_subscriber)

    realsense_subscriber.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()

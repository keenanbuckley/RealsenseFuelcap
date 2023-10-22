import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge, CvBridgeError

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class RealsenseSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_subscriber')

        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_rect_raw',
            self.listener_callback,
            10)
        self.subscription


    def listener_callback(self, image):
        self.get_logger().info(f'Recieved image of shape: {image.width}x{image.height}')

def main(args=None):
    rclpy.init(args=args)
    realsense_subscriber = RealsenseSubscriber()

    rclpy.spin(realsense_subscriber)

    realsense_subscriber.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()

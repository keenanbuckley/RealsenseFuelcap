import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class Realsense_Simulated_Collection(Node):
    def __init__(self):
        super().__init__('realsense_subscriber')
        self.bridge = CvBridge()

        # Update the topic names based on your simulated camera's topics
        self.color_subscription = self.create_subscription(
            Image,
            '/demo_cam/mycamera/image_demo',
            self.color_listener_callback,
            10)
        self.depth_subscription = self.create_subscription(
            Image,
            '/demo_cam/mycamera/depth_demo',
            self.depth_listener_callback,
            10)

        self.color_img_msg = None
        self.depth_img_msg = None

    def keyboard_listener(self):
        try:
            image_color = self.bridge.imgmsg_to_cv2(self.color_img_msg, desired_encoding="bgr8")
            image_depth = self.bridge.imgmsg_to_cv2(self.depth_img_msg, desired_encoding="16UC1")
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        image_depth = np.array(image_depth, dtype=np.uint16)

        # Save color image
        cv2.imwrite('saved_img.png', image_color)

        # Print depth info
        self.get_logger().info(f"Depth Max: {np.max(image_depth)}, Depth Min: {np.min(image_depth)}")
        self.get_logger().info(f"Depth Shape: {image_depth.shape[0]}x{image_depth.shape[1]}")
        self.get_logger().info(f"Depth at center: {image_depth[int(image_depth.shape[0]/2), int(image_depth.shape[1]/2)]}")

    def color_listener_callback(self, img_msg):
        self.color_img_msg = img_msg
        self.get_logger().info(f'Image of shape: {self.color_img_msg.width}x{self.color_img_msg.height}')
    
    def depth_listener_callback(self, img_msg):
        self.depth_img_msg = img_msg

def main(args=None):
    rclpy.init(args=args)
    realsense_subscriber = Realsense_Simulated_Collection()

    try:
        rclpy.spin(realsense_subscriber)
    except KeyboardInterrupt:
        realsense_subscriber.keyboard_listener()

    realsense_subscriber.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

class RealsenseSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_subscriber')
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/color/image_rect_raw',
            self.color_listener_callback,
            10)
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self.depth_listener_callback,
            10)
        self.color_subscription
        self.depth_subscription
        #keyboard.on_release_key('g', self.keyboard_listener)
        self.color_img_msg = None
        self.depth_img_msg = None

    def keyboard_listener(self, other=None):
        bridge = CvBridge()

        try:
            image_color = bridge.imgmsg_to_cv2(self.color_img_msg, desired_encoding="passthrough")
            image_depth = bridge.imgmsg_to_cv2(self.depth_img_msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
        
        image_depth = np.array(image_depth, dtype=np.float32)
        
        image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
        cv2.imwrite('saved_color.png', image_color)

        print(np.max(image_depth), np.min(image_depth))
        print(image_depth.shape[0], image_depth.shape[1])
        print(len(image_depth[0]))
        print(image_depth[int(image_depth.shape[0]/2), int(image_depth.shape[1]/2)])

    def color_listener_callback(self, img_msg):
        self.color_img_msg = img_msg
        self.get_logger().info(f'Image of shape: {self.color_img_msg.width}x{self.color_img_msg.height}')
    
    def depth_listener_callback(self, img_msg):
        self.depth_img_msg = img_msg
       

def main(args=None):
    rclpy.init(args=args)

    realsense_subscriber = RealsenseSubscriber()

    try:
        rclpy.spin(realsense_subscriber)
    except KeyboardInterrupt:
        realsense_subscriber.keyboard_listener()

    realsense_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
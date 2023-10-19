import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
#import keyboard

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
        image_bgr = np.array(self.color_img_msg.data, dtype=np.uint8).reshape(self.color_img_msg.height, self.color_img_msg.width, -1)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        cv2.imwrite('saved_rgb.png', image_rgb)
        
        image_depth = np.array(self.depth_img_msg.data, dtype=np.uint8).reshape(self.color_img_msg.height, self.color_img_msg.width, -1)
        print(image_depth.shape)
        #image_rgb = cv2.cvtColor(image_depth, cv2.COLOR_BGR2RGB)
        #cv2.imwrite('saved_depth.png', image_rgb)

        #imagebgr = cv2.cvtColor(imagebgr, cv2.COLOR_GRAY2BGR)
        #imagergb = cv2.cvtColor(imagebgr, cv2.COLOR_BGR2RGB)
        cv2.imwrite('saved_depth0.png', image_depth[:,:,0])
        cv2.imwrite('saved_depth1.png', image_depth[:,:,1])

        print(np.max(image_depth), np.min(image_depth))
        image_depth = np.bitwise_or(np.left_shift(np.array(image_depth[:,:,1], dtype=np.uint16), 8), np.array(image_depth[:,:,0], dtype=np.uint16))
        
        cv2.imwrite('saved_depth.png', image_depth)

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
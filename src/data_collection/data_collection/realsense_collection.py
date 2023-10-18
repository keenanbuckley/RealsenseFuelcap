import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
#import keyboard

class RealsenseSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_rect_raw',
            self.listener_callback,
            10)
        self.subscription
        #keyboard.on_release_key('g', self.keyboard_listener)
        self.image = None

    def keyboard_listener(self, other=None):
        self.get_logger().info(f'Image of shape: {self.image.width}x{self.image.height}')
        print(self.image.data[0][0])

    def listener_callback(self, image):
        self.image = image

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
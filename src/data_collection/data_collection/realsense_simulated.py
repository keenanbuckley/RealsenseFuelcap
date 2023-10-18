import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from sensor_msgs.msg import Image

class SimulatedRealsense(Node):

    def __init__(self):
        super().__init__('simulated_realsense')
        # Create a publisher with a QoS profile suitable for image streams
        qos = QoSProfile(depth=10)
        self.publisher = self.create_publisher(Image, '/camera/color/image_rect_raw', qos)
        
        # Create a timer to periodically publish images
        self.timer = self.create_timer(0.1, self.publish_image)  # 10 Hz

    def publish_image(self):
        # Create a dummy image message
        img = Image()
        img.header.stamp = self.get_clock().now().to_msg()
        img.height = 480  # example height
        img.width = 640   # example width
        img.encoding = 'rgb8'
        img.step = img.width * 3  # 3 bytes per pixel for rgb8
        img.data = [255] * (img.width * img.height * 3)  # white image    
        # Publish the image
        self.get_logger().info(f'Publishing Image')
        self.publisher.publish(img)

def main(args=None):
    rclpy.init(args=args)

    realsense_subscriber = SimulatedRealsense()

    rclpy.spin(realsense_subscriber)

    realsense_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

from sensor_msgs.msg import Image

class DisplayPng(Node):

    def __init__(self):
        super().__init__('simulated_realsense')
        # Create a publisher with a QoS profile suitable for image streams
        qos = QoSProfile(depth=10)
        self.publisher = self.create_publisher(Image, '/camera/color/image_rect_raw', qos)
        
        # Create a timer to periodically publish images
        self.timer = self.create_timer(0.1, self.publish_image)  # 10 Hz

    def publish_image(self):
        img = Image()

        # Read the PNG image from file
        image_path = '/~/mines_ws/RealsenseFuelcap/src/data_collection/data_collection/displayRviz/exampleImage.png'
        cv_image = cv2.imread(image_path)

        # Convert the OpenCV image to a ROS Image message
        bridge = CvBridge()
        img_msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

        # Update the image message metadata
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = 'bgr8'  # Assuming the image is in BGR format
        img_msg.step = cv_image.shape[1] * 3  # 3 bytes per pixel for BGR8

        # Publish the image
        self.get_logger().info(f'Publishing Image')
        self.publisher.publish(img_msg)
      
def main(args=None):
    rclpy.init(args=args)

    realsense_subscriber = SimulatedRealsense()

    rclpy.spin(realsense_subscriber)

    realsense_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

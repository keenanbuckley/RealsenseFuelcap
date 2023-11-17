import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from PIL import Image as PILImage
import io

class ImagePublisher(Node):

    def __init__(self):
        super().__init__('image_publisher')
        self.publisher = self.create_publisher(Image, '/camera/color/image_rect_raw', 10)
        self.timer = self.create_timer(1.0, self.publish_image)  # Adjust the timer duration as needed

    def publish_image(self):
        # Load the image using PIL
        pil_image = PILImage.open('exampleImage.jpeg')

        print("Image format:", pil_image.format)
        print("Image mode:", pil_image.mode)
        print("Image size:", pil_image.size)

        # Convert the PIL image to a ROS2 Image message
        img_msg = Image()
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.height = pil_image.height
        img_msg.width = pil_image.width
        print("Image height:", pil_image.height)
        print("Image width:", pil_image.width)
        img_msg.encoding = 'rgb8'  # Adjust if your image has a different encoding
        img_msg.step = img_msg.width * 3  # Assuming 3 channels for an RGB image

        # Convert the PIL image to bytes and assign to the Image message data field
        byte_io = io.BytesIO()
        pil_image.save(byte_io, format='PNG')
        img_msg.data = byte_io.getvalue()

        # Publish the Image message
        self.publisher.publish(img_msg)
        self.get_logger().info('Image published')

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)
    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

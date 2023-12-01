import rclpy
from rclpy.node import Node

from custom_interfaces.msg import FuelCapDetectionInfo
from geometry_msgs.msg import PoseStamped

class DetectionInfoSubscriber(Node):
    def __init__(self):
        super().__init__('detection_info_subscriber')

        self.detection_info_subscription = self.create_subscription(
            FuelCapDetectionInfo,
            '/fuelcap_detection_info',
            self.detection_info_callback,
            10)
    
    def detection_info_callback(self, info_msg: FuelCapDetectionInfo):
        if info_msg.is_fuelcap_detected:
            log_msg = 'recieved detection info:\n'
            log_msg += f'Inference Time: {info_msg.inference_time}\n'
            log_msg += f'Position: {info_msg.pose_stamped.pose.position.x}, {info_msg.pose_stamped.pose.position.y}, {info_msg.pose_stamped.pose.position.z}\n'
            log_msg += f'Orientation: {info_msg.pose_stamped.pose.orientation.x}, {info_msg.pose_stamped.pose.orientation.y}, {info_msg.pose_stamped.pose.orientation.z}, {info_msg.pose_stamped.pose.orientation.w}\n'
            log_msg += f'Bounding Box Confidence: {info_msg.bbox_confidence_score}\n'
            log_msg += f'Additional Info: {info_msg.detection_info}\n'
            self.get_logger().info(log_msg)
        else:
            self.get_logger().error(f'No fuelcap detected: {info_msg.detection_info}')

def main(args=None):
    rclpy.init(args=args)
    detection_node = DetectionInfoSubscriber()

    try:
        rclpy.spin(detection_node)
    except SystemExit:
        pass

    detection_node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
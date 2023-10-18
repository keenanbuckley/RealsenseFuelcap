#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pcl
from sensor_msgs.msg import PointCloud2

class CapturePointCloud(Node):

    def __init__(self):
        super().__init__('capture_point_cloud')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/demo_cam/mycamera/points_demo',  # topic name
            self.point_cloud_callback,
            1  # QoS profile depth (set to 1 to capture a single message)
        )
        self.captured = False

    def point_cloud_callback(self, msg):
        if not self.captured:
            # Convert the PointCloud2 message to a PCL point cloud
            point_cloud = pcl.PointCloud()
            pcl_msg = pcl.create_cloud_xyz32(point_cloud.header, msg.data)

            # Save the PCL point cloud to a PCD file
            pcl.save(pcl_msg, 'captured_point_cloud.pcd')
            self.get_logger().info('Point cloud captured and saved as captured_point_cloud.pcd')
            self.captured = True

def main(args=None):
    rclpy.init(args=args)
    node = CapturePointCloud()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

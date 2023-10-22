#!/usr/bin/env python3

import rclpy
import numpy as np
import cv2
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py.point_cloud2 import read_points
from cv_bridge import CvBridge

class CapturePointCloud(Node):

    def __init__(self):
        super().__init__('capture_point_cloud')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/demo_cam/mycamera/points_demo',
            self.point_cloud_callback,
            1
        )
        self.rgb_subscription = self.create_subscription(
            Image,
            '/demo_cam/mycamera/rgb_image',  # replace with your RGB topic name
            self.rgb_callback,
            1
        )
        self.bridge = CvBridge()
        self.rgb_image = None
        self.captured = False

    def rgb_callback(self, msg):
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def point_cloud_callback(self, msg):
        if not self.captured and self.rgb_image is not None:
            # Extract points from PointCloud2 message
            points = np.array(list(read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))

            # Create an empty depth image
            depth_img = np.zeros((480, 640))  # Assuming a resolution of 640x480

            # Camera intrinsics (replace with your actual values)
            fx, fy = 525.0, 525.0  # focal lengths
            cx, cy = 320.0, 240.0  # optical center

            for p in points:
                x, y, z = p
                if z == 0:  # Avoid division by zero
                    continue
                u = int((x * fx) / z + cx)
                v = int((y * fy) / z + cy)
                if 0 <= u < 640 and 0 <= v < 480:
                    depth_img[v, u] = z

            # Normalize depth values to 0-255 range
            depth_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
            depth_img = np.uint8(depth_img)

            # Apply colormap to produce a heatmap representation
            heatmap_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)

            # Blend the heatmap with the RGB image
            blended_img = cv2.addWeighted(self.rgb_image, 0.6, heatmap_img, 0.4, 0)

            # Save blended image
            cv2.imwrite('blended_image.png', blended_img)
            self.get_logger().info('Blended image saved as blended_image.png')
            self.captured = True

def main(args=None):
    rclpy.init(args=args)
    node = CapturePointCloud()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

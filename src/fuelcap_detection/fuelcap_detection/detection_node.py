import cv2
import numpy as np
import rclpy
import time
import sys

from keypoints_detection.keypoint_model import KPModel
from bounding_box.bounding_box import BBoxModel
from image_transformations.coordinate_transforms import *

import PIL

from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from custom_interfaces.srv import CaptureImage
from custom_interfaces.msg import FuelCapDetectionInfo

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from rclpy.parameter import ParameterMsg, ParameterType, ParameterValue

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header

class DetectionNode(Node):
    def __init__(self, bbox_model: BBoxModel, kp_model: KPModel, exposure: int = None, enable_annotations=False):
        super().__init__('fuelcap_detection')
        self.bridge = CvBridge()
        self.enable_annotations = enable_annotations

        if not exposure is None:
            # Parameters to automatically change for the realsense node
            camera_parameters = list()
            camera_parameters.append(ParameterMsg(name='depth_module.exposure', value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=exposure)))
            
            # Wait until realsense node is active, then change its parameters
            msg = SetParameters.Request()
            msg.parameters = camera_parameters
            self.cli = self.create_client(SetParameters, '/camera/camera/set_parameters')
            while not self.cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info('realsense service not available, waiting again...')
            self.cli.call_async(msg)

        # Create subscribers to the realsense color and depth topics
        self.color_subscription = self.create_subscription(
            Image,
            '/camera/color/image_rect_raw',
            self.color_listener_callback,
            10)
        self.depth_subscription = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.depth_listener_callback,
            10)

        # Create publisher for detection info
        self.detection_info_publisher = self.create_publisher(FuelCapDetectionInfo, 'fuelcap_detection_info', 10)
        
        # Create publisher for annotated images
        if self.enable_annotations:
            self.annotated_image_publisher = self.create_publisher(Image, 'annotated_image', 10)
        
        # For storing image messages while waiting for the other to arrive
        self.color_img_msg = None
        self.depth_img_msg = None

        # Initialize the bounding box and keypoints DL models
        self.bbox_model = bbox_model
        self.kp_model = kp_model
        self.K = IntrinsicsMatrix()
        self.pose_msg = PoseStamped()
    
    def color_listener_callback(self, img_msg: Image):
        self.color_img_msg = img_msg
        if not self.depth_img_msg is None:
            self.rgbd_callback()
            self.color_img_msg = None
            self.depth_img_msg = None
    
    def depth_listener_callback(self, img_msg: Image):
        self.depth_img_msg = img_msg
        if not self.color_img_msg is None:
            self.rgbd_callback()
            self.color_img_msg = None
            self.depth_img_msg = None
    
    def rgbd_callback(self):
        inference_start_time = time.time()
        try:
            image_color = self.bridge.imgmsg_to_cv2(self.color_img_msg, desired_encoding="bgr8")
            image_depth = self.bridge.imgmsg_to_cv2(self.depth_img_msg, desired_encoding="16UC1")
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")
            return
        
        image_depth = np.array(image_depth, dtype=np.uint16)
        pil_image = PIL.Image.fromarray(image_color)
        bbox, score = self.bbox_model.find_bbox(pil_image)
        if not bbox is None:
            kpts = self.kp_model.predict(image_color, bbox)
            rotation, translation, _, _ = self.kp_model.predict_position(self.K, image_depth, 12)
            bbox = [round(x) for x in bbox.tolist()]
            cv2.rectangle(image_color, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,255,0), 2)
            if not translation is None and not rotation is None:
                H = TransformationMatrix(R=rotation, t=translation)
                annotate_img(image_color, H, self.K)
                position, orientation = H.as_pos_and_quat()
                self.pose_msg.pose.position.x = float(position[0])
                self.pose_msg.pose.position.y = float(position[1])
                self.pose_msg.pose.position.z = float(position[2])
                self.pose_msg.pose.orientation.x = float(orientation[0])
                self.pose_msg.pose.orientation.y = float(orientation[1])
                self.pose_msg.pose.orientation.z = float(orientation[2])
                self.pose_msg.pose.orientation.w = float(orientation[3])
                # TODO: set header for pose stamped
                self.pose_msg.header = Header(stamp=self.color_img_msg.header.stamp, frame_id='base_link')
                self.publish_detection_info(inference_start_time, score, True, "")
            else:
                self.get_logger().error(f"Could not calculate position")
                self.publish_detection_info(inference_start_time, score, False, "Could not calculate position")
                self.kp_model.reset_positions()
        else:
            self.get_logger().error(f"No BBox detected")
            self.publish_detection_info(inference_start_time, 0, False, "No BBox detected")
            self.kp_model.reset_positions()
        if self.enable_annotations:
            self.publish_annotated_image(image_color)
    
    def publish_detection_info(self, inference_start_time, bbox_score, is_fuelcap_detected, detection_info):
        msg = FuelCapDetectionInfo()
        msg.inference_time = float(time.time() - inference_start_time)
        msg.bbox_confidence_score = float(bbox_score)
        msg.is_fuelcap_detected = bool(is_fuelcap_detected)
        msg.detection_info = str(detection_info)
        msg.pose_stamped = self.pose_msg
        self.detection_info_publisher.publish(msg)

    def publish_annotated_image(self, annotated_image):
        msg = self.bridge.cv2_to_imgmsg(np.array(annotated_image), "bgr8")
        self.annotated_image_publisher.publish(msg)
        self.get_logger().info(f'publishing annotated image')
    
def main(args=None):
    py_args = sys.argv[1:]

    try: 
        exposure = int(py_args[0])
        print(f'Exposure set to {exposure}')
    except:
        print("No exposure detected")
        exposure = None
    
    bbox_model = BBoxModel("models/bbox_model.pth")
    kp_model = KPModel("models/keypoint_checkpoint.pt")

    rclpy.init(args=args)
    detection_node = DetectionNode(bbox_model, kp_model, exposure=exposure, enable_annotations=True)

    try:
        rclpy.spin(detection_node)
    except SystemExit:
        pass

    detection_node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
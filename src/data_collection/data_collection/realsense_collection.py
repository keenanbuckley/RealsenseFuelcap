import cv2
import numpy as np
import rclpy
import time
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from custom_interfaces.srv import CaptureImage

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from rclpy.parameter import Parameter, ParameterMsg, ParameterType, ParameterValue

import sys
import os

class RealsenseCollection(Node):
    def __init__(self, exposure: int = 7500,  n_pictures: int = 1, rate: int = 1):
        super().__init__('realsense_subscriber')
        self.bridge = CvBridge()
        self.n_pictures = n_pictures
        self.n_recieved = 0
        self.rate = rate
        timer_period = 1

        parameters = list()
        #parameters.append(ParameterMsg(name='depth_module.enable_auto_exposure', value=ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=True)))
        #parameters.append(ParameterMsg(name='depth_module.auto_exposure_roi.top', value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=65)))
        #parameters.append(ParameterMsg(name='depth_module.auto_exposure_roi.bottom', value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=720-65)))
        #parameters.append(ParameterMsg(name='depth_module.auto_exposure_roi.left', value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=75)))
        #parameters.append(ParameterMsg(name='depth_module.auto_exposure_roi.right', value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=1280-75)))
        parameters.append(ParameterMsg(name='depth_module.exposure', value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=exposure)))

        msg = SetParameters.Request()
        #print(msg.get_fields_and_field_types())
        msg.parameters = parameters

        self.cli = self.create_client(SetParameters, '/camera/camera/set_parameters')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.cli.call_async(msg)

        # Update the topic names based on your simulated camera's topics
        # self.timer = self.create_timer(timer_period, self.periodic)
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
        self.srv = self.create_service(CaptureImage, '/capture_image', self.capture_image_callback)

        self.frame_id = ""
        self.color_img_msg = None
        self.depth_img_msg = None
        self._new_color_img_msg = None
        self._new_depth_img_msg = None
    
    def capture_image_callback(self, request, response):
        path = request.path
        timestamp = int(1000*time.time())
        self.get_logger().info(f'{int(self.n_recieved / self.rate)} of shape: {self.color_img_msg.width}x{self.color_img_msg.height}')
        self.get_logger().info(f'frame_id: color {self.color_img_msg.header.stamp}, depth {self.depth_img_msg.header.stamp}')

        try:
            image_color = self.bridge.imgmsg_to_cv2(self.color_img_msg, desired_encoding="bgr8")
            image_depth = self.bridge.imgmsg_to_cv2(self.depth_img_msg, desired_encoding="16UC1")
        except CvBridgeError as e:
            self.get_logger().error(f"Error converting image: {e}")
            return
        
        # Convert depth image to np array
        image_depth = np.array(image_depth, dtype=np.uint16)

        # Save color image
        if not os.path.exists(f'{path}/color'):
            os.makedirs(f'{path}/color')
        cv2.imwrite(f'{path}/color/{timestamp}.png', image_color)

        # Save depth image
        if not os.path.exists(f'{path}/depth'):
            os.makedirs(f'{path}/depth')
        np.save(f'{path}/depth/{timestamp}.npy', image_depth)

        # Log the images in image_list.txt
        with open(f'{path}/image_list.txt', 'a') as f:
            f.write(f'{timestamp}\n')

        # Print depth info
        #self.get_logger().info(f"Depth Max: {np.max(image_depth)}, Depth Min: {np.min(image_depth)}")
        #self.get_logger().info(f"Depth Shape: {image_depth.shape[0]}x{image_depth.shape[1]}")
        #self.get_logger().info(f"Depth at center: {image_depth[int(image_depth.shape[0]/2), int(image_depth.shape[1]/2)]}")
        
        response.image_name = str(timestamp)
        return response

    def color_listener_callback(self, img_msg: Image):
        self._new_color_img_msg = img_msg
        if not self._new_depth_img_msg is None:
            self.color_img_msg = self._new_color_img_msg
            self.depth_img_msg = self._new_depth_img_msg
            self._new_color_img_msg = None
            self._new_depth_img_msg = None
    
    def depth_listener_callback(self, img_msg):
        self._new_depth_img_msg = img_msg
        if not self._new_color_img_msg is None:
            self.color_img_msg = self._new_color_img_msg
            self.depth_img_msg = self._new_depth_img_msg
            self._new_color_img_msg = None
            self._new_depth_img_msg = None


def main(args=None):
    py_args = sys.argv[1:]

    try: 
        exposure = int(py_args[0])
        print(f'Exposure set to {exposure}')
    except:
        print("No exposure detected, setting to 7500")
        exposure = 7500
    
    '''
    try:
        n = int(py_args[1])
        print(f"Collecting {n} pictures")
    except:
        print("No arguments detected, collecting 1000 pictures")
        n = 1

    try: 
        rate = int(py_args[2])
        print(f'Rate set to {rate}')
    except:
        print("No rate detected, setting to 1")
        rate = 1
    '''

    rclpy.init(args=args)
    realsense_subscriber = RealsenseCollection(exposure=exposure)

    try:
        rclpy.spin(realsense_subscriber)
    except EOFError:
        pass

    realsense_subscriber.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()

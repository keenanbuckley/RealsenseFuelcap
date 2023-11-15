source /opt/ros/humble/setup.bash
source install/setup.bash

#Starts real world camera
alias start_realsense="cd ~/mines_ws && ros2 launch realsense2_camera rs_launch.py depth_module.profile:=1280x720x30 enable_rgbd:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true depth_module.enable_auto_exposure:=true"
#Simulates real world camera
alias start_sim="cd ~/mines_ws && source install/setup.bash && ros2 run data_collection simulated"
#Builds all ROS2 packages
alias ros_build="cd ~/mines_ws && colcon build --packages-select custom_interfaces data_collection && source install/setup.bash"
#Start data collection
alias camera_launch="cd ~/mines_ws && source install/setup.bash && ros2 run data_collection camera"
#Gazebo Setup --For debugging, Should be done when image is built
alias gazebo_Setup='sudo apt-get install ros-humble-gazebo-plugins && source /usr/share/gazebo/setup.sh'
#Start Gazebo simulation
alias gazebo_Start='source /usr/share/gazebo/setup.sh && gazebo --verbose --ros-args --params-file $(ros2 pkg prefix gazebo_plugins)/share/gazebo_plugins/worlds/gazebo_ros_depth_camera_demo.world'
#Start Gazebo data collection
alias gazebo_camera_launch="cd ~/mines_ws && source install/setup.bash && ros2 run data_collection gazebo_camera"
#Save an image to the data directory
alias capture_image="ros2 service call /capture_image custom_interfaces/srv/CaptureImage \"{path: 'data'}\""
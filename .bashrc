source /opt/ros/humble/setup.bash

alias start_realsense="cd ~/mines_ws && ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true"
alias start_sim="cd ~/mines_ws && source install/setup.bash && ros2 run data_collection simulated"
alias ros_build="cd ~/mines_ws && colcon build --packages-select data_collection"
alias camera_launch="cd ~/mines_ws && source install/setup.bash && ros2 run data_collection camera"
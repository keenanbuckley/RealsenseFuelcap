source /opt/ros/humble/setup.bash

alias start_realsense="cd ~/mines_ws && ros2 launch realsense2_camera rs_launch.py depth_module.profile:=1280x720x30 enable_rgbd:=true colorizer.enable:=true enable_sync:=true align_depth.enable:=true enable_color:=true enable_depth:=true"
alias start_sim="cd ~/mines_ws && source install/setup.bash && ros2 run data_collection simulated"
alias ros_build="cd ~/mines_ws && colcon build --packages-select data_collection"
alias camera_launch="cd ~/mines_ws && source install/setup.bash && ros2 run data_collection camera"
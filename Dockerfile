FROM osrf/ros:humble-desktop-full

# install ubuntu packages
RUN apt update && apt install -y \
        chrony \
        curl \
        git \
        sudo \
        ssh \
        tmux \
        vim \
        xterm

# install ros packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
        ros-humble-pcl-conversions

ENV HOME /home/mines
WORKDIR "/home/mines/mines_ws"

ENV DISPLAY=host.docker.internal:0.0

CMD ["/bin/bash"]
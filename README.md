# Stratom FuelCap POS Estimation Project Documentation

## Setup

### ROS Node
1. Clone the project from GitHub: [RealsenseFuelcap](https://github.com/keenanbuckley/RealsenseFuelcap)
2. Open the VSCode terminal in the project directory.
3. Build the Cuda docker container using the following command:
   ```
   docker buildx build -t mines_cuda -f Dockerfile.CUDA .
   ```

### Unity Data Collection
For information on Unity Data Collection, please refer to the following GitHub repository: [KeypointSimulation](https://github.com/jschauer1/KeypointSimulation).

## Dependencies

### ROS Node
- CUDA
- PyTorch CUDA
- VScode
- Docker

## Building ROS


Once all dependencies are met, you can run the code using the following procedure

### Linux

- build and run the docker file. To do so execute the following command:
   ```
      xhost +local:root 
      docker run -it --runtime=nvidia --net host --ipc host -e "DISPLAY=$DISPLAY" \
         -v "$HOME/.Xauthority:/root/.Xauthority:ro" --privileged \
         -v $HOME/path/to/repository:/home/mines/mines_ws \
         --rm --name cuda_container mines_cuda
   ```
- source the bashfile
   ```
      source .bashrc
   ```

- Build the ROS workspace by running the alias:
   ```
      ros_build
   ```


## Running Nodes and services
The bashrc file includes aliases for several functonalities within the code, including starting the realsense camera, data collection, and fuelcap detection nodes, calling the capture image service, and displaying the fuelcap detection info message

### Linux
Follow the steps for building ROS, after building ros, in any unused terminal within the container execute the command below, then start your node:
```
source .bashrc
```


#### Nodes
For the following nodes, execute the corresponding alias

1. Realsense Node
   ```
   start_realsense
   ```
2. Data Collection Node
   ```
   data_collection_launch
   ```
3. Fuelcap Detection Node
   ```
   fuelcap_detection_launch
   ```
4. Display Detection Info
   ```
   detection_info_launch
   ```
#### Services
To call the capture image service, run the following command:
```
capture_image
```

## Contributing

(Existing content or instructions on how to contribute to the project)

## Contributors

[Keenan Buckley](https://github.com/HFocus) - <https://github.com/HFocus>

## License

This project uses a permissive BSD Zero-Clause License. For more information, see the accompanying [LICENSE](/LICENSE) file.

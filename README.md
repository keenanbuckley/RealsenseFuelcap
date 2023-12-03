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

## Contributing

(Existing content or instructions on how to contribute to the project)

## Contributors

[Keenan Buckley](https://github.com/HFocus) - <https://github.com/HFocus>

## License

This project uses a permissive BSD Zero-Clause License. For more information, see the accompanying [LICENSE](/LICENSE) file.

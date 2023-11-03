import shutil
import os
import sys
import random

def move_first_file(src_directory, dst_directory, number_of_samples):
    """
    Moves the first file found in the src_directory to the dst_directory.
    
    Parameters:
    - src_directory: A string, the path to the source directory.
    - dst_directory: A string, the path to the destination directory.
    """
    # Check if the source directory exists
    if not os.path.isdir(src_directory):
        print(f"The directory {src_directory} does not exist.")
        return
    
    # Check if the destination directory exists, if not, create it
    if not os.path.exists(dst_directory):
        os.makedirs(dst_directory)
        print(f"The directory {dst_directory} did not exist, created now.")
    
    # List all files in the source directory
    files = os.listdir(src_directory)
    if not files:
        print(f"No files found in the directory {src_directory}.")
        return
    # Get the first file
    for i in range(0,number_of_samples):
        random_integer = random.randint(0, len(files)-1)  # Will give a random integer from 1 to 10
        first_file = files[random_integer]
        files.pop(random_integer)
        src_file_path = os.path.join(src_directory, first_file)

        # Move the file
        try:
            # Define the destination file path
            dst_file_path = os.path.join(dst_directory, first_file)
            
            # Move the file
            shutil.move(src_file_path, dst_file_path)
            print(f"Moved {first_file} to {dst_directory} number {i}")
        except Exception as e:
            print(f"An error occurred while moving {src_file_path}: {e}")

# Example usage:
source_directory = os.path.expanduser('~/mines_ws/--Original file path}')  # Replace with your source directory path
destination_dir = os.path.expanduser('~/mines_ws/TestNewLoc')  # Replace with your destination directory path
if len(sys.argv) > 1:
    try:
        number_of_samples = int(sys.argv[1])
        print(f"The number of samples specified is {number_of_samples}.")
    except ValueError:
        print(f"Please provide a valid integer. '{sys.argv[1]}' is not an integer.")
else:
    print("This script requires an integer argument representing the number of samples.")


move_first_file(source_directory, destination_dir,number_of_samples)

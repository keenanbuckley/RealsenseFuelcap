import cv2
import numpy as np

def normalize_and_save_image(input_image_path, output_image_path):
    # Read the image using cv2
    image = cv2.imread(input_image_path)
    
    # Normalize the image to be in the range [0, 1]
    normalized_image = image.astype('float32') / 255.0
    
    # Save the normalized image to the specific filepath as a .npy file
    np.save(output_image_path, normalized_image)
    
    return normalized_image
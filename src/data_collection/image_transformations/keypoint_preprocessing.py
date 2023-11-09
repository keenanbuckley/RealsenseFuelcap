import numpy as np
import json, cv2
import time


filepath = "src/data_collection/image_transformations/data/"

def assemble_image(file: str, bounding_box =[(580,215),(870,500)], min_depth: int = 70, max_depth: int = 500, out_width = 100, method: str = "resize"):
    """
    Assembles image data for keypoints model. Crops only image defiend by bounding box, resizes it and concatonates deptha and
    color data into one array
    Args:
    file (string): filename
    bounding_box (list of coordinates): pair of coordinates defining the boinding box
    min_depth (int): minimum depth in mm
    max_depth (int): maximum depth in mm
    out_width (int): out put dimension of the image 
    method (str): method for resizing bounding box, if resize, stretch image, if "expand" expand bounding box to be a square then shrink
    """
    assert method in ["resize", "expand"]



    depth_data = np.int16(np.load(f"{filepath}{file}.npy"))
    color_data = cv2.imread(f"{filepath}{file}.png")

    (x1,y1),(x2,y2) = bounding_box

    if method == "expand":
        width = x2 - x1
        height = y2 - y1
        xc = int(np.mean(x2, x1))
        yc = int(np.mean(y2, y1))
        new_dim = max(width, height)

        y2 = yc + new_dim // 2
        y1 = yc - new_dim // 2
        x2 = xc + new_dim // 2
        x1 = xc - new_dim // 2



    depth_data = depth_data[y1:y2, x1:x2]
    color_data = color_data[y1:y2, x1:x2]

    color_resize = cv2.resize(color_data, (out_width, out_width))
    depth_resize = cv2.resize(depth_data, (out_width, out_width))

    depth_mask = np.logical_and(depth_resize < max_depth, depth_resize >= min_depth)
    depth_trim = np.where(depth_mask, depth_resize, min_depth)

    depth_norm = (np.float16(depth_trim) - min_depth) / (max_depth - min_depth)
    color_norm = np.float32(color_resize) / 255

    assembled_data = np.concatenate((color_norm, depth_norm[:,:,np.newaxis]), axis=2)

    return assembled_data




def main():
    file = "image"
    t0 = time.time()
    img = assemble_image(file)
    time_elapes = time.time() - t0
    print(f"Elapsed time: {time_elapes}")

    cv2.imshow("Color_data", img[:,:,:3])
    cv2.imshow("Depth data", img[:,:,3])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # np.save(f"{filepath}{file}.npy", np.ones(dim, dtype=np.int16))
    # cv2.imwrite(f"{filepath}{file}.jpg", np.zeros(dim, dtype=np.uint8))




if __name__ == "__main__":
    main()

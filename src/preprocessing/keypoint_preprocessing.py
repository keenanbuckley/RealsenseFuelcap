import numpy as np
import json, cv2
import time, os





def assemble_image(
        color: np.ndarray,
        depth: np.ndarray,
        bounding_box, 
        min_depth: int = 70, max_depth: int = 500,
        out_width = 100, 
        method: str = "resize", 
        bbox_frmt="xyxy", 
        filepath="data/"
        ):
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
    assert bbox_frmt in ["xywh", "xyxy"]

    # depth_data = np.int16(np.load(f"{filepath}/depth/{file}.npy"))
    # color_data = cv2.imread(f"{filepath}/color/{file}.png")

    assert depth.shape[:2] == color.shape[:2]

    if bbox_frmt == "xyxy":
        x1,y1,x2,y2 = bounding_box
        width, height = x2-x1, y2-y1
    elif bbox_frmt == "xywh":
        x1,y1,width,height = bounding_box
        x2,y2=x1+width,y1+height
    else:
        raise Exception

    if method == "expand":
        xc = int(np.mean([x2, x1]))
        yc = int(np.mean([y2, y1]))
        new_dim = max(width, height)

        y2 = yc + new_dim // 2
        y1 = yc - new_dim // 2
        x2 = xc + new_dim // 2
        x1 = xc - new_dim // 2

    depth = depth[y1:y2, x1:x2]
    color = color[y1:y2, x1:x2]

    color_resize = cv2.resize(color, (out_width, out_width))
    depth_resize = cv2.resize(depth, (out_width, out_width))

    depth_mask = np.logical_and(depth_resize < max_depth, depth_resize >= min_depth)
    depth_trim = np.where(depth_mask, depth_resize, min_depth)

    depth_norm = (np.float32(depth_trim) - min_depth) / (max_depth - min_depth)
    color_norm = np.float32(color_resize) / 255

    assembled_data = np.concatenate((color_norm, depth_norm[:,:,np.newaxis]), axis=2)

    return assembled_data


def normalize_image(image_path):
    image = cv2.imread(image_path)
    normalized_image = image.astype('float32') / 255.0
    return normalized_image 


def clean_data():
    with open("L:\stratom\RealsenseFuelcap\src\data_collection\data_reading\combined_data\main.json", 'r') as f:
        data = json.load(f) 

    files = data.keys()

    new_data = {}
    invalid_data = {}
    for file in files:
        file_data = data[file]

        if "Keypoint" not in file_data.keys() or "Boundingbox" not in file_data.keys():
            print("Not enough data for", file, file_data.keys())
            file_data['reason'] = "missing key"
            invalid_data[file] = file_data
            continue
        

        file_bbox = file_data["Boundingbox"]
    
        if isinstance(file_bbox[0], list):
            file_bbox = file_bbox[0]
    
        file_bbox = file_bbox[:2] + [file_bbox[i] + file_bbox[i+2] for i in range(2)]
        file_bbox = [round(i) for i in file_bbox]

        keypoints = file_data["Keypoint"]
        

        if len(keypoints) > 0 and isinstance(keypoints[0], list):
            keypoints = keypoints[0]

        if not len(keypoints) == 20:
            print("Not enough keypoints in", file, len(keypoints))
            file_data['reason'] = "not enough keypoints"
            invalid_data[file] = file_data
            continue

        keypoints = [round(i) for i in keypoints]

        new_data[file] = {"bbox": file_bbox, "keypoints": keypoints}

    remap_keys = {"Keypoint": "keypoints", "Boundingbox": "bbox", "reason": "reason"}
    invalid_data = {file: {remap_keys[old_key]: value for old_key, value in file_data.items()} for file, file_data in invalid_data.items()}

    bbox_str = json.dumps(new_data, indent=4).replace('[\n\t','[').replace(",\n            ", ", ").replace("[\n            ", '[').replace("\n        ]", "]")
    with open("L:\stratom\RealsenseFuelcap\src\data_collection\data_reading\combined_data\\new_data.json",'w') as f:
        f.write(bbox_str)

        
    data_str = json.dumps(invalid_data, indent=4)
    with open("L:\stratom\RealsenseFuelcap\src\data_collection\data_reading\combined_data\\bad_data.json",'w') as f:
        f.write(data_str)

    print(f"There are {len(invalid_data.keys())} images missing values")


 


def main():
    # clean_data()
    pass

   

if __name__ == "__main__":
    main()

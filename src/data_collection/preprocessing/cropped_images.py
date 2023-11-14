import json, cv2, numpy as np
from keypoint_preprocessing import assemble_image

data_path = "data"


def main():
    img_data_path = f"{data_path}/all_data.json"

    with open(img_data_path, 'r') as f:
        data = json.load(f)

    img_names = data.keys()
    for imfile in img_names:
        color = cv2.imread(f"{data_path}/color/{imfile}.png")
        depth = np.load(f"{data_path}/depth/{imfile}.npy")
        bbox = data[imfile]['bbox']
        img = assemble_image(color, depth, bbox)
        np.save(f"{data_path}/cropped_images/{imfile}_cropped.npy", img)


if __name__ == "__main__":
    main()
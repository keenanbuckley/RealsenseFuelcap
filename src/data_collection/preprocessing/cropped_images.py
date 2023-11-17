import json, cv2, numpy as np
from keypoint_preprocessing import assemble_image

data_path = "data"


def main():
    img_data_path = f"{data_path}/all_data.json"
    out_data_path = f"{data_path}/all_data_cropped.json"
    
    out_data = {}
    out_width = 256
    with open(img_data_path, 'r') as f:
        data = json.load(f)

    img_names = data.keys()
    for imfile in img_names:
        color = cv2.imread(f"{data_path}/color/{imfile}.png")
        depth = np.load(f"{data_path}/depth/{imfile}.npy")
        
        keypoints = data[imfile]['keypoints']
        bbox = data[imfile]['bbox']

        crop_keypoints = []
        for i in range(len(keypoints) // 2):
            x = keypoints[2*i] - bbox[0]
            y = keypoints[2*i+1] - bbox[1]

            original_width, original_height = [bbox[i+2] - bbox[i] for i in range(2)]
            out_x = out_width * (x / original_width)
            out_y = out_width * (y / original_height)

            crop_keypoints = [out_x, out_y, 1]

        img = assemble_image(color, depth, bbox, out_width=out_width)
        np.save(f"{data_path}/cropped_images/{imfile}_cropped.npy", img)

        cropped_img_dict = {}
        cropped_img_dict["bbox"] = [0, 0, out_width, out_width]
        cropped_img_dict['keypoints'] = crop_keypoints
        out_data[imfile] = cropped_img_dict
    
    out_str = json.dumps(out_data, indent=4).replace('[\n\t','[').replace(",\n            ", ", ").replace("[\n            ", '[').replace("\n        ]", "]")
    with open(out_data_path,'w') as f:
        f.write(out_str)



if __name__ == "__main__":
    main()
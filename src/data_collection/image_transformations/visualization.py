import pandas as pd
import cv2, numpy as np
from coordinate_transforms import TransformationMatrix, IntrinsicsMatrix, calculate_matrix, annotate_img

if __name__ == "__main__":
        
    df = pd.read_csv("data/GroundTruth/fuelcap_data.csv")
    # print(df.head())
    for idx, row in df.iterrows():
        if idx == 0:
            continue
        name = row["image_name"]
        x = row["dX"]
        y = row["dY"]
        z = row["dZ"] - (25 / 25.4)
        angle_mount = row["angle_mount"]
        angle_cap = row["angle_cap"]
        
        H = calculate_matrix(x,y,z,angle_mount=-angle_mount, angle_cap=angle_cap)
        K = IntrinsicsMatrix()
        print(name)
        img = cv2.imread(f"data/GroundTruth/color/{name}.png")

        img = annotate_img(img, H, K, axis_len=30)
        print(f"Displaying {name}: ({x}, {y}, {z}), mount={angle_mount}, cap={angle_cap}")
        cv2.imshow("Annotated Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
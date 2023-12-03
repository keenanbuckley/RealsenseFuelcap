import pandas as pd
import cv2, numpy as np
from coordinate_transforms import TransformationMatrix, IntrinsicsMatrix, calculate_matrix, annotate_img

if __name__ == "__main__":
    """Show annotations for ground truth data
    """
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
        
        H_u_min = calculate_matrix(x-0.25,y,z,angle_mount=(angle_mount), angle_cap=angle_cap, units='in')
        H_v_min = calculate_matrix(x,y-0.25,z,angle_mount=(angle_mount), angle_cap=angle_cap, units='in')
        H_v_max = calculate_matrix(x,y+0.25,z,angle_mount=(angle_mount), angle_cap=angle_cap, units='in')
        H_u_max = calculate_matrix(x+0.25,y,z,angle_mount=(angle_mount), angle_cap=angle_cap, units='in')
        H = calculate_matrix(x,y,z,angle_mount=angle_mount, angle_cap=angle_cap, units='in')
        K = IntrinsicsMatrix()
        img = cv2.imread(f"data/GroundTruth/color/{name}.png")

        cv2.rectangle(img,(img.shape[1]//2,0),(img.shape[1]//2,img.shape[0]),(0,0,0), thickness=2)
        cv2.rectangle(img,(0,img.shape[0]//2),(img.shape[1],img.shape[0]//2),(0,0,0), thickness=2)
        cv2.circle(img,(img.shape[1]//2,img.shape[0]//2), 6, (0,0,0), -1)
        cv2.circle(img,(img.shape[1]//2,img.shape[0]//2), 3, (255,255,255), -1)
        #img = annotate_img(img, H_u_min, K, axis_len=5)
        #img = annotate_img(img, H_v_min, K, axis_len=5)
        #img = annotate_img(img, H_u_max, K, axis_len=5)
        #img = annotate_img(img, H_v_max, K, axis_len=5)
        img = annotate_img(img, H, K, axis_len=30)
        print(f"Displaying {name}: ({x}, {y}, {z}), mount={angle_mount}, cap={angle_cap}")
        print(H.as_pos_and_quat()[0])
        cv2.imshow("Annotated Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
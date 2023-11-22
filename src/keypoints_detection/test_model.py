import cv2, numpy as np


img = cv2.imread("data/NewData/sim_20231116162554797.png")
blurr = cv2.GaussianBlur(img, (7,7), 10)
cv2.imshow("Image", img)
cv2.imshow("Blurred", blurr)
cv2.waitKey(0)
cv2.destroyAllWindows()
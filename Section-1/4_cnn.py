import cv2
import numpy as np


img = cv2.imread("lenna.png")

kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

img2 = cv2.filter2D(img, cv2.CV_8U, kernel)

cv2.imshow("image", img)
cv2.imshow("image2", img2)

cv2.waitKey(0)

cv2.destroyAllWindows()
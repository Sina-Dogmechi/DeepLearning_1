import cv2
import numpy as np

# Gray
# img = cv2.imread("lenna.png", 0)

img = cv2.imread("lenna.png")

kernel = np.array([[-1, 1]])

kernel2 = np.array([[-1],
                    [1]])

img2 = cv2.filter2D(img, cv2.CV_8U, kernel)
img3 = cv2.filter2D(img, cv2.CV_8U, kernel2)

cv2.imshow("image", img)
cv2.imshow("image2", img2)
cv2.imshow("image3", img3)

cv2.waitKey(0)

cv2.destroyAllWindows()
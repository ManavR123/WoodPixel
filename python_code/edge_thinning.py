import cv2
import numpy as np

img = cv2.imread("alan_edges.png", 0)
img = 255 - img
cv2.imshow("original", img)
img1 = img.copy()

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
thin = np.zeros(img.shape, dtype="uint8")

while cv2.countNonZero(img1) != 0:
    erode = cv2.erode(img1, kernel)
    opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
    subset = erode - opening
    thin = cv2.bitwise_or(subset, thin)
    img1 = erode.copy()

cv2.imwrite("alan_edges_thin.png", thin)

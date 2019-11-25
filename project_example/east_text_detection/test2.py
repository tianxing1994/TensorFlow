import cv2 as cv
import numpy as np


rect1 = np.array([[10, 10], [60, 10], [60, 60], [10, 60]])
rect2 = np.array([[40, 40], [90, 40], [90, 90], [40, 90]])
falg, vertices = cv.rotatedRectangleIntersection(rect1, rect2)
print(vertices)

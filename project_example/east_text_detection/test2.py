import cv2 as cv
import numpy as np
from shapely.geometry import Polygon


rect1 = np.array([[10, 10], [60, 10], [60, 60], [10, 60]])

ploygon1 = Polygon(rect1)
print(ploygon1.exterior)
print(type(ploygon1.exterior))

print(ploygon1.interiors)
print(type(ploygon1.interiors))

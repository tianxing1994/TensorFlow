import numpy as np
from imutils.object_detection import non_max_suppression


boxes = np.array([[1, 1, 4, 4],
                  [1.1, 0.9, 4.2, 3.5],
                  [3.6, 3.9, 5.4, 6.2]])


probs = np.array([1, 0.7, 0.6])

result = non_max_suppression(boxes=boxes, probs=probs, overlapThresh=0.95)

print(result)








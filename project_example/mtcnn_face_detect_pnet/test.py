import os
import numpy as np
import cv2 as cv


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


image_data = ['9--Press_Conference/9_Press_Conference_Press_Conference_9_770.jpg', 1, [(207, 174, 570, 631)]]

image_relative_path, face_count, bounding_box = image_data
image_path = os.path.join("dataset/WIDER_train", image_relative_path)
image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
m, n = image.shape
roi_min_size = 12
roi_max_size = np.min((m, n))
print(roi_max_size)
result = list()
for i in range(100):
    data = list()
    roi_size = np.random.randint(roi_min_size, roi_max_size)
    x = np.random.randint(n - roi_size)
    y = np.random.randint(m - roi_size)
    roi = image[y:y + roi_size, x:x + roi_size]
    selected_bounding_box = (x, y, roi_size, roi_size)
    print(x, y)
    show_image(roi)

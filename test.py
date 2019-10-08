import numpy as np
import cv2 as cv


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


image_path = 'project_example/mtcnn_face_detect_pnet/dataset/WIDER_train/0--Parade/0_Parade_marchingband_1_849.jpg'

image = cv.imread(image_path)
print(image.shape)

# 449 330 122 149 0 0 0 0 0 0
cv.cv2.rectangle(image, (449, 330), (449+122, 330+149), (0, 255, 0), 2)

show_image(image)
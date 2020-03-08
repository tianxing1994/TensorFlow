import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def generate_training_set(n_images, image_size, min_object_size, max_object_size):
    images = np.zeros(shape=(n_images, image_size, image_size, 1))
    bounding_boxes = np.zeros(shape=(n_images, 4))
    for i in range(n_images):
        width, height = np.random.randint(min_object_size, max_object_size, size=2)
        x = np.random.randint(0, image_size - width)
        y = np.random.randint(0, image_size - height)
        images[i, y:y+height, x: x+width] = 1.0
        bounding_boxes[i] = [x, y, width, height]

    return images, bounding_boxes


def demo1():
    n_images = 500
    image_size = 16
    min_object_size = 1
    max_object_size = 8

    images, bounding_boxes = generate_training_set(n_images, image_size, min_object_size, max_object_size)
    print("images shape: ", images.shape)
    print("bounding boxes shape: ", bounding_boxes.shape)

    for image, bounding_box in zip(images, bounding_boxes):
        print(bounding_box)
        show_image(image)

    return


if __name__ == '__main__':
    demo1()

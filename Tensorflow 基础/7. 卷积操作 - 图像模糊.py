import tensorflow as tf
import cv2 as cv
import numpy as np


def show_image(image):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    """模糊图像, 相当于是均值模糊"""
    image = cv.imread(r'C:\Users\Administrator\PycharmProjects\TensorFlow\datasets\cat.jpg')
    image = np.array(image, dtype=np.float32)
    shape = image.shape
    image = np.expand_dims(image, axis=0)

    filter = tf.Variable(tf.ones([7, 7, 3, 1]))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        res = tf.nn.conv2d(image, filter, strides=[1, 2, 2, 1], padding="SAME")
        res_image = sess.run(tf.reshape(res, (int(shape[0] / 2), int(shape[1] / 2))))
        res_image = res_image / res_image.max()

        show_image(res_image)

    return


def demo2():
    """卷积核改为 (11, 11), 模糊图像, 相当于是均值模糊"""
    image = cv.imread(r'C:\Users\Administrator\PycharmProjects\TensorFlow\datasets\cat.jpg')
    image = np.array(image, dtype=np.float32)
    shape = image.shape
    image = np.expand_dims(image, axis=0)

    filter = tf.Variable(tf.ones([11, 11, 3, 1]))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        res = tf.nn.conv2d(image, filter, strides=[1, 2, 2, 1], padding="SAME")
        res_image = sess.run(tf.reshape(res, (int(shape[0] / 2), int(shape[1] / 2))))
        res_image = res_image / res_image.max()

        show_image(res_image)

    return


if __name__ == '__main__':
    demo2()

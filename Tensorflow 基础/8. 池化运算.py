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
    """池化运算演示."""
    data = tf.constant([[[3, 2, 3, 4],
                         [2, 6, 2, 4],
                         [1, 2, 1, 5],
                         [4, 3, 2, 1]]])

    data = tf.reshape(data, [1, 4, 4, 1])
    max_pooling = tf.nn.max_pool(data, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

    with tf.Session() as sess:
        print(sess.run(max_pooling))

    return


def demo2():
    """当有多个通道时, 池化运算演示: 当有多个通道时, 每个通道的池化运算是独立的. """
    data = tf.constant([[[[3, 5],
                          [3, 6],
                          [1, 5],
                          [5, 7]],

                         [[5, 2],
                          [6, 4],
                          [4, 1],
                          [6, 3]],

                         [[1, 3],
                          [4, 6],
                          [6, 4],
                          [4, 3]],

                         [[3, 6],
                          [4, 3],
                          [4, 6],
                          [6, 7]]]])

    data = tf.reshape(data, [1, 4, 4, 2])
    max_pooling = tf.nn.max_pool(data, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")

    with tf.Session() as sess:
        print(sess.run(max_pooling))

    return


def demo3():
    image = cv.imread(r'C:\Users\Administrator\PycharmProjects\TensorFlow\datasets\cat.jpg')
    shape = image.shape
    image = np.array(image, np.float32)
    image = np.expand_dims(image, axis=0)

    filter = tf.Variable(tf.ones([7, 7, 3, 1]))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        res = tf.nn.conv2d(image, filter, strides=[1, 2, 2, 1], padding="SAME")
        res = tf.nn.max_pool(res, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID")
        res_image = sess.run(tf.reshape(res, (int(shape[0]/4), int(shape[1]/4))))
        res_image = res_image / res_image.max()
        show_image(res_image)
    return


if __name__ == '__main__':
    demo2()

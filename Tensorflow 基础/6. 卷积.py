import cv2 as cv
import numpy as np
import tensorflow as tf


def show_image(image):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    image_path = r'C:\Users\Administrator\PycharmProjects\TensorFlow\datasets\cat.jpg'
    image = cv.imread(image_path)

    image = np.expand_dims(image, axis=0)
    print(image.shape)

    input = tf.constant(image, dtype=tf.float64)
    filter_nd = np.array([[[[1], [1], [1]],
                           [[1], [1], [1]],
                           [[1], [1], [1]]],

                          [[[1], [1], [1]],
                           [[1], [1], [1]],
                           [[1], [1], [1]]],

                          [[[1], [1], [1]],
                           [[1], [1], [1]],
                           [[1], [1], [1]]]], np.int)
    print(filter_nd.shape)

    filter = tf.constant(filter_nd, dtype=tf.float64)

    conv2d = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        image_blurred = sess.run(conv2d)
        image_blurred = image_blurred / image_blurred.max()
        image_blurred = image_blurred.astype(np.float)
        image_blurred = np.squeeze(image_blurred)

        show_image(image_blurred)
    return


def demo2():
    image_path = r'C:\Users\Administrator\PycharmProjects\TensorFlow\datasets\cat.jpg'
    image = cv.imread(image_path, flags=cv.IMREAD_GRAYSCALE)

    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)
    print(image.shape)

    input = tf.constant(image, dtype=tf.float64)
    filter_nd = np.array([[[[1]],
                           [[1]],
                           [[1]]],

                          [[[1]],
                           [[1]],
                           [[1]]],

                          [[[1]],
                           [[1]],
                           [[1]]]], np.int)
    print(filter_nd.shape)

    filter = tf.constant(filter_nd, dtype=tf.float64)

    conv2d = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        image_blurred = sess.run(conv2d)
        image_blurred = image_blurred / image_blurred.max()
        image_blurred = image_blurred.astype(np.float)
        image_blurred = np.squeeze(image_blurred)
        show_image(image_blurred)

    return


def demo3():
    """
    当 input 输入是多个通道时, filter 需要具有同样多的通道, 其卷积后的值, 是三个通道的和.
    :return:
    """
    input_nd = np.array([[[[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]],

                          [[9, 8, 7],
                           [6, 5, 4],
                           [3, 2, 1]],

                          [[9, 8, 7],
                           [6, 5, 4],
                           [3, 2, 1]]]], dtype=np.int)

    print(input_nd.shape)

    filter_nd = np.array([[[[1], [1], [1]],
                           [[1], [1], [1]],
                           [[1], [1], [1]]],

                          [[[1], [1], [1]],
                           [[1], [1], [1]],
                           [[1], [1], [1]]],

                          [[[1], [1], [1]],
                           [[1], [1], [1]],
                           [[1], [1], [1]]]], np.int)

    print(filter_nd.shape)
    input = tf.constant(input_nd, dtype=tf.float64)
    filter = tf.constant(filter_nd, dtype=tf.float64)
    conv2d = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        output = sess.run(conv2d)
        print(output)
        print(output.shape)

    print((1+2+3+4+5+6+7+8+9)*3)
    print((1+2+3+4+5+6) + (9+8+7+6+5+4))
    print((1+2+3+4+5+6) + (9+8+7+6+5+4) + (9+8+7+6+5+4))

    return


def demo4():
    """
    卷积核 filter (height, width, in_channel, out_channel),
    其 out_channel 维度的值不同, 相当于多个卷积核, 分别输出一张卷积后的数组.
    :return:
    """
    input_nd = np.array([[[[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]],

                          [[9, 8, 7],
                           [6, 5, 4],
                           [3, 2, 1]],

                          [[9, 8, 7],
                           [6, 5, 4],
                           [3, 2, 1]]]], dtype=np.int)

    print(input_nd.shape)

    filter_nd = np.array([[[[1, 2], [1, 2], [1, 2]],
                           [[1, 2], [1, 2], [1, 2]],
                           [[1, 2], [1, 2], [1, 2]]],

                          [[[1, 2], [1, 2], [1, 2]],
                           [[1, 2], [1, 2], [1, 2]],
                           [[1, 2], [1, 2], [1, 2]]],

                          [[[1, 2], [1, 2], [1, 2]],
                           [[1, 2], [1, 2], [1, 2]],
                           [[1, 2], [1, 2], [1, 2]]]], np.int)

    print(filter_nd.shape)
    input = tf.constant(input_nd, dtype=tf.float64)
    filter = tf.constant(filter_nd, dtype=tf.float64)
    conv2d = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        output = sess.run(conv2d)
        print(output)
        print(output.shape)

    return


if __name__ == '__main__':
    demo4()


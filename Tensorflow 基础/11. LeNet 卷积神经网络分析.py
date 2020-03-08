"""
卷积: 我们知道, 向量的点乘为: a·b = a1b1+a2b2+a3b3+...+anbn, 这其实就是卷积的计算方法.
又: 向量的点乘(内积), 相当于是把一个向量投影到另一个向量上, 相当于改变坐标系来观察原向量.
我们知道, 卷积过程中, 卷积核的值是不变的. 这就相当于把图像每个区域投影到另一个坐标系中进行观察.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2 as cv
import time
import os
import struct
import numpy as np


def show_image(image):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    """
    读取 mnist 数据库.
    这里, 还是不知道应该怎么去查看和理解这个数据库.
    :return:
    """
    mnist_data_set = input_data.read_data_sets("MNIST_data", one_hot=True)

    batch_xs, batch_ys = mnist_data_set.train.next_batch(1)

    print(batch_xs.shape)
    print(batch_ys.shape)

    # 显示图片
    image = batch_xs.reshape(28, 28)
    show_image(image)

    batch_xs, batch_ys = mnist_data_set.test.next_batch(1)

    print(batch_xs.shape)
    print(batch_ys.shape)

    # 样本的数量
    num_examples = mnist_data_set.train.num_examples
    print(num_examples)
    return


def demo2():
    """
    卷积执行后, 每一个输出通道的值不相等.
    :return:
    """
    mnist_data_set = input_data.read_data_sets("MNIST_data", one_hot=True)
    batch_xs, batch_ys = mnist_data_set.train.next_batch(1)
    x_image = batch_xs.reshape(1, 28, 28, 1)
    image = tf.constant(x_image)

    checkpoint_path = "C:/Users/Administrator/PycharmProjects/TensorFlow/datasets/models/LeNet_model/model.ckpt"
    reader = tf.train.NewCheckpointReader(checkpoint_path)

    filter1 = reader.get_tensor("filter1")
    bias1 = reader.get_tensor("bias1")
    conv1 = tf.nn.conv2d(image, filter1, strides=[1, 1, 1, 1], padding="SAME")
    h_conv1 = tf.nn.sigmoid(conv1 + bias1)

    with tf.Session() as sess:
        h_conv1_image = sess.run(h_conv1)
        h_conv1_image_0 = h_conv1_image[:, :, :, 0]
        h_conv1_image_1 = h_conv1_image[:, :, :, 1]
        result = h_conv1_image_0 - h_conv1_image_1
        print(result)
        # h_conv1_image = np.reshape(h_conv1_image, (28, 28))
        # show_image(h_conv1_image)

    return


def demo3():
    mnist_data_set = input_data.read_data_sets("MNIST_data", one_hot=True)
    batch_xs, batch_ys = mnist_data_set.train.next_batch(1)
    x_image = batch_xs.reshape(1, 28, 28, 1)
    image = tf.constant(x_image)

    checkpoint_path = "C:/Users/Administrator/PycharmProjects/TensorFlow/datasets/models/LeNet_model/model.ckpt"
    reader = tf.train.NewCheckpointReader(checkpoint_path)

    filter1 = reader.get_tensor("filter1")
    bias1 = reader.get_tensor("bias1")
    conv1 = tf.nn.conv2d(image, filter1, strides=[1, 1, 1, 1], padding="SAME")
    h_conv1 = tf.nn.sigmoid(conv1 + bias1)

    max_pool2 = tf.nn.max_pool(h_conv1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="max_pool2")

    filter2 = reader.get_tensor("filter2")
    bias2 = reader.get_tensor("bias2")
    conv2 = tf.nn.conv2d(max_pool2, filter2, strides=(1, 1, 1, 1), padding="SAME", name="conv2")
    h_conv2 = tf.nn.sigmoid(conv2 + bias2, name="h_conv2")

    max_pool3 = tf.nn.max_pool(h_conv2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="max_pool3")

    filter3 = reader.get_tensor("filter3")
    bias3 = reader.get_tensor("bias3")
    conv3 = tf.nn.conv2d(max_pool3, filter3, strides=(1, 1, 1, 1), padding="SAME", name="conv3")
    h_conv3 = tf.nn.sigmoid(conv3 + bias3, name="h_conv3")



    with tf.Session() as sess:
        h_conv1_image = sess.run(h_conv1)
        h_conv1_image = h_conv1_image[:, :, :, 0]
        h_conv1_image = np.reshape(h_conv1_image, (28, 28))
        show_image(h_conv1_image)

        max_pool2_image = sess.run(max_pool2)
        max_pool2_image = max_pool2_image[:, :, :, 0]
        max_pool2_image = np.reshape(max_pool2_image, (14, 14))
        show_image(max_pool2_image)

        h_conv2_image = sess.run(h_conv2)
        h_conv2_image = h_conv2_image[:, :, :, 0]
        h_conv2_image = np.reshape(h_conv2_image, (14, 14))
        show_image(h_conv2_image)

        max_pool3_image = sess.run(max_pool3)
        max_pool3_image = max_pool3_image[:, :, :, 0]
        max_pool3_image = np.reshape(max_pool3_image, (7, 7))
        show_image(max_pool3_image)

        h_conv3_image = sess.run(h_conv3)
        h_conv3_image = h_conv3_image[:, :, :, 0]
        h_conv3_image = np.reshape(h_conv3_image, (7, 7))
        show_image(h_conv3_image)
    return


if __name__ == '__main__':
    demo2()

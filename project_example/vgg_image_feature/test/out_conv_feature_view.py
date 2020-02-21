# -*- coding: utf-8 -*-
# https://blog.csdn.net/weixin_41521681/article/details/87884646
import scipy.io
import numpy as np
import tensorflow as tf
import cv2 as cv
import scipy.misc


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def conv_layer(input, weights, biases):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding="SAME")
    return tf.nn.bias_add(conv, biases)


def pooling_layer(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel


def imread(path):
    return cv.imread(path).astype(np.float)


def net(data_path, x_placeholder):
    layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
              'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
              'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
              'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
              'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4')

    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    x = x_placeholder
    for i, name in enumerate(layers):
        print('i=', i)
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            x = conv_layer(x, kernels, bias)
        elif kind == 'relu':
            x = tf.nn.relu(x)
        elif kind == 'pool':
            x = pooling_layer(x)
    return x, mean_pixel


if __name__ == '__main__':
    vgg_model_path = '../../../datasets/models/vgg/vgg_scipy/imagenet-vgg-verydeep-19.mat'
    image_path = '../../../datasets/cat.jpg'
    image = cv.imread(image_path).astype(np.float)
    image = cv.resize(image, dsize=(224, 224))
    shape = (1, image.shape[0], image.shape[1], image.shape[2])
    x_placeholder = tf.placeholder('float', shape=shape)
    x, mean_pixel = net(vgg_model_path, x_placeholder)
    image_p = np.array([preprocess(image, mean_pixel)])

    with tf.Session() as sess:
        features = x.eval(feed_dict={x_placeholder: image_p})
        print('shape of features is ', features.shape)
        ret = np.abs(features[0, :, :, 0])
        ret /= np.max(ret)
        show_image(ret)

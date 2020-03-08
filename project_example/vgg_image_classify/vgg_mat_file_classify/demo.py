# -*- coding: utf-8 -*-
"""
Allocation of 411041792 exceeds 10% of system memory.
内存溢出.

虽然存在内存溢出. 但分类效果还是不错.


ConvNet_19

images -> [batch, height, width, channels] -> [None×224×224×3]

[name]             [input]                            [output]                         [action]        [output_size]                       [conv kernel size]    [filters]             [stride]    [pad]
conv1_1,            input=images,                      output=conv1_1,                  convolution,    output_size=None×224×224×64,        kernel_size=3×3,      filters=64,           stride=1,   SAME,
relu1_1,            input=conv1_1,                     output=relu1_1,                  relu,           output_size=None×224×224×64,
conv1_2,            input=relu1_1,                     output=conv1_2,                  convolution,    output_size=None×224×224×64,        kernel_size=3×3,      filters=64,           stride=1,   SAME,
relu1_2,            input=conv1_2,                     output=relu1_2,                  relu,           output_size=None×224×224×64,
pool1,              input=relu1_2,                     output=pool1,                    max_pool,       output_size=None×112×112×64,        kernel_size=2×2                             stride=2,   SAME,

conv2_1,            input=pool1,                       output=conv2_1,                  convolution,    output_size=None×112×112×128,       kernel_size=3×3,      filters=128,          stride=1,   SAME,
relu2_1,            input=conv2_1,                     output=relu2_1,                  relu,           output_size=None×112×112×128,
conv2_2,            input=relu2_1,                     output=conv2_2,                  convolution,    output_size=None×112×112×128,       kernel_size=3×3,      filters=128,          stride=1,   SAME,
relu2_2,            input=conv2_2,                     output=relu2_2,                  relu,           output_size=None×112×112×128,
pool2,              input=relu2_2,                     output=pool2,                    max_pool,       output_size=None×56×56×128,         kernel_size=2×2                             stride=2,   SAME,

conv3_1,            input=pool2,                       output=conv3_1,                  convolution,    output_size=None×56×56×256,         kernel_size=3×3,      filters=256,          stride=1,   SAME,
relu3_1,            input=conv3_1,                     output=relu3_1,                  relu,           output_size=None×56×56×256,
conv3_2,            input=relu3_1,                     output=conv3_2,                  convolution,    output_size=None×56×56×256,         kernel_size=3×3,      filters=256,          stride=1,   SAME,
relu3_2,            input=conv3_2,                     output=relu3_2,                  relu,           output_size=None×56×56×256,
conv3_3,            input=relu3_2,                     output=conv3_3,                  convolution,    output_size=None×56×56×256,         kernel_size=3×3,      filters=256,          stride=1,   SAME,
relu3_3,            input=conv3_3,                     output=relu3_3,                  relu,           output_size=None×56×56×256,
conv3_4,            input=relu3_3,                     output=conv3_4,                  convolution,    output_size=None×56×56×256,         kernel_size=3×3,      filters=256,          stride=1,   SAME,
relu3_4,            input=conv3_4,                     output=relu3_4,                  relu,           output_size=None×56×56×256,
pool3,              input=relu3_4,                     output=pool3,                    max_pool,       output_size=None×28×28×256,         kernel_size=2×2                             stride=2,   SAME,

conv4_1,            input=pool3,                       output=conv4_1,                  convolution,    output_size=None×28×28×512,         kernel_size=3×3,      filters=512,          stride=1,   SAME,
relu4_1,            input=conv4_1,                     output=relu4_1,                  relu,           output_size=None×28×28×512,
conv4_2,            input=relu4_1,                     output=conv4_2,                  convolution,    output_size=None×28×28×512,         kernel_size=3×3,      filters=512,          stride=1,   SAME,
relu4_2,            input=conv4_2,                     output=relu4_2,                  relu,           output_size=None×28×28×512,
conv4_3,            input=relu4_2,                     output=conv4_3,                  convolution,    output_size=None×28×28×512,         kernel_size=3×3,      filters=512,          stride=1,   SAME,
relu4_3,            input=conv4_3,                     output=relu4_3,                  relu,           output_size=None×28×28×512,
conv4_4,            input=relu4_3,                     output=conv4_4,                  convolution,    output_size=None×28×28×512,         kernel_size=3×3,      filters=512,          stride=1,   SAME,
relu4_4,            input=conv4_4,                     output=relu4_4,                  relu,           output_size=None×28×28×512,
pool4,              input=relu4_4,                     output=pool4,                    max_pool,       output_size=None×14×14×512,         kernel_size=2×2                             stride=2,   SAME,

conv5_1,            input=pool4,                       output=conv5_1,                  convolution,    output_size=None×14×14×512,         kernel_size=3×3,      filters=512,          stride=1,   SAME,
relu5_1,            input=conv5_1,                     output=relu5_1,                  relu,           output_size=None×14×14×512,
conv5_2,            input=relu5_1,                     output=conv5_2,                  convolution,    output_size=None×14×14×512,         kernel_size=3×3,      filters=512,          stride=1,   SAME,
relu5_2,            input=conv5_2,                     output=relu5_2,                  relu,           output_size=None×14×14×512,
conv5_3,            input=relu5_2,                     output=conv5_3,                  convolution,    output_size=None×14×14×512,         kernel_size=3×3,      filters=512,          stride=1,   SAME,
relu5_3,            input=conv5_3,                     output=relu5_3,                  relu,           output_size=None×14×14×512,
conv5_4,            input=relu5_3,                     output=conv5_4,                  convolution,    output_size=None×14×14×512,         kernel_size=3×3,      filters=512,          stride=1,   SAME,
relu5_4,            input=conv5_4,                     output=relu5_4,                  relu,           output_size=None×14×14×512,
pool5,              input=relu5_4,                     output=pool5,                    max_pool,       output_size=None×7×7×512,           kernel_size=2×2                             stride=2,   SAME,

full_connect_6,     input=pool5,                       output=full_connect_6,           convolution,    output_size=None×1×1×4096,          kernel_size=7×7       filters=4096,         stride=1,   SAME,
relu6,              input=full_connect_6,              output=relu6,                    relu,           output_size=None×1×1×4096,
full_connect_7,     input=relu6,                       output=full_connect_7,           convolution,    output_size=None×1×1×4096,          kernel_size=1×1       filters=4096,         stride=1,   SAME,
relu7,              input=full_connect_7,              output=relu7,                    relu,           output_size=None×1×1×4096,
full_connect_8,     input=relu7,                       output=full_connect_8,           convolution,    output_size=None×1×1×1000,          kernel_size=1×1       filters=1000,         stride=1,   SAME,
prob                input=full_connect_8,              output=prob,                     softmax,        output_size=None×1×1×1000,

"""
import scipy.io
import numpy as np
import tensorflow as tf
import cv2 as cv
import scipy.misc
import glob


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def conv_layer(input, weights, biases, padding):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1), padding=padding)
    return tf.nn.bias_add(conv, biases)


def pooling_layer(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel


def imread(path):
    return cv.imread(path).astype(np.float)


def net(param_data, x_placeholder):
    layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
              'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
              'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
              'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
              'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
              'full_connect_6', 'relu6',
              'full_connect_7', 'relu7',
              'full_connect_8', 'prob')

    weights = param_data['layers'][0]
    x = x_placeholder
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            params, pad, operation_type, operation_name, stride = weights[i][0][0]
            kernels, bias = params[0]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            x = conv_layer(x, kernels, bias, "SAME")
            print(f'i = {i}, '
                  f'x shape = {x.shape}, '
                  f'kernels shape: {kernels.shape}, '
                  f'bias shape: {bias.shape}, '
                  f'pad: {pad[0]}, '
                  f'type: {operation_type[0]}, '
                  f'name: {operation_name[0]}, '
                  f'stride: {stride[0]}')
        elif kind == 'relu':
            x = tf.nn.relu(x)
            operation_type, operation_name = weights[i][0][0]
            print(f'i = {i}, '
                  f'x shape = {x.shape}, '
                  f'type: {operation_type[0]}, '
                  f'name: {operation_name[0]}')
        elif kind == 'pool':
            x = pooling_layer(x)
            operation_name, window_shape, pad, operation_type, pool_type, stride = weights[i][0][0]
            print(f'i = {i}, '
                  f'x shape = {x.shape}, '
                  f'name: {operation_name[0]}, '
                  f'window_shape: {window_shape[0]}, '
                  f'pad: {pad[0]}, '
                  f'type: {operation_type[0]}, '
                  f'pool_type: {pool_type[0]}, '
                  f'stride: {stride[0]}')
        elif kind == 'full':
            params, pad, operation_type, operation_name, stride = weights[i][0][0]
            kernels, bias = params[0]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            x = conv_layer(x, kernels, bias, "VALID")
            print(f'i = {i}, '
                  f'x shape = {x.shape}, '
                  f'kernels shape: {kernels.shape}, '
                  f'bias shape: {bias.shape}, '
                  f'pad: {pad[0]}, '
                  f'type: {operation_type[0]}, '
                  f'name: {operation_name[0]}, '
                  f'stride: {stride[0]}')
        elif kind == 'prob':
            x = tf.nn.softmax(x, axis=3)
            operation_type, operation_name = weights[i][0][0]
            print(f'i = {i}, '
                  f'x shape = {x.shape}, '
                  f'type: {operation_type[0]}, '
                  f'name: {operation_name[0]}')
    return x


if __name__ == '__main__':
    vgg_model_path = '../../../datasets/models/vgg/vgg_scipy/imagenet-vgg-verydeep-19.mat'
    param_data = scipy.io.loadmat(vgg_model_path)
    mean = param_data['normalization'][0][0][0]
    x_placeholder = tf.placeholder('float', shape=(1, 224, 224, 3))
    x = net(param_data, x_placeholder)

    with tf.Session() as sess:
        image_path_list = glob.glob("../../../datasets/image/*")
        for image_path in image_path_list:
            print(image_path)
            image = cv.imread(image_path).astype(np.float)
            image = cv.resize(image, dsize=(224, 224))
            image_p = np.array([image - mean], dtype=np.float64)

            prob = x.eval(feed_dict={x_placeholder: image_p})

            index = int(np.squeeze(np.argmax(prob, axis=3)))
            print(f'index: {index}. ')
            cls0, cls1 = param_data['classes'][0][0]
            result = cls1[0][index]
            print(f'class: {result}')

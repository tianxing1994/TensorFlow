import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.models.research.slim.nets import vgg

from classical_net.vgg.vgg16.vgg16_license_plates_detect.load_data import batch_load_data, show_image


slim = tf.contrib.slim


def vgg16_conv(inputs, reuse=None, scope='vgg_16', fc_conv_padding='VALID'):
    with tf.variable_scope(
            scope, 'vgg_16', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'

        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
            return net


def inference(inputs):
    with slim.arg_scope(vgg.vgg_arg_scope()):
        net = vgg16_conv(inputs)

        with tf.variable_scope('migration'):
            net = slim.conv2d(net, 4096, [1, 1], scope='fc1')
            net = slim.conv2d(net, 1024, [1, 1], scope='fc2')
            net = slim.conv2d(net, 5, [1, 1], activation_fn=None, scope='fc3')
            net = tf.squeeze(net, [1, 2])
            net = tf.nn.softmax(net)
    return net



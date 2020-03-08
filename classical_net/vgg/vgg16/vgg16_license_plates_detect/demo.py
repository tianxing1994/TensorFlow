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


inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='inputs')
y_true = tf.placeholder(tf.float32, [None, 4], name='y_true')

with slim.arg_scope(vgg.vgg_arg_scope()):
    net = vgg16_conv(inputs)

    with tf.variable_scope('migration'):
        net = slim.conv2d(net, 4096, [1, 1], scope='fc1')
        net = slim.conv2d(net, 2048, [1, 1], scope='fc2')
        net = slim.conv2d(net, 1024, [1, 1], scope='fc3')
        net = slim.conv2d(net, 4, [1, 1], activation_fn=None, scope='fc4')
        y_pred = tf.squeeze(net, [1, 2])

restore_var_list = [var for var in tf.trainable_variables() if var.name.startswith("vgg_16")]
optimize_var_list = [var for var in tf.trainable_variables() if var.name.startswith("migration")]


def demo1():
    # 训练
    ckpt_path = '../../../../datasets/classical_model/vgg16/vgg_16.ckpt'

    saver1 = tf.train.Saver(var_list=restore_var_list, max_to_keep=3)
    saver2 = tf.train.Saver(max_to_keep=3)

    loss = tf.reduce_mean(tf.square(y_true - y_pred))
    op_train = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, var_list=optimize_var_list)

    n_epoch = 100
    batch_size = 32

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver1.restore(sess, ckpt_path)

        for epoch in range(n_epoch):
            train_loss, n_batch = 0, 0

            for x_train, y_train in batch_load_data(batch_size=batch_size, flag="train"):
                theloss, _ = sess.run([loss, op_train], feed_dict={inputs: x_train, y_true: y_train})
                train_loss += theloss
                print(f"temp: theloss {theloss}")
                n_batch += 1
            print(f"train loss: {train_loss / n_batch}")
            saver2.save(sess, "model/box_regression.ckpt", global_step=epoch + 1)
    return


def demo2():
    # 查看训练后的效果.
    ckpt_path = tf.train.latest_checkpoint('model')
    saver = tf.train.Saver(max_to_keep=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_path)

        for x_train, _ in batch_load_data(batch_size=1, flag="train"):
            y_ = sess.run(y_pred, feed_dict={inputs: x_train})
            y_ = np.squeeze(y_, axis=0) * 224
            x1, y1, x2, y2 = y_[0], y_[1], y_[2], y_[3]
            print(x1, y1, x2, y2)
            image = np.array(x_train[0], dtype=np.uint8)
            image = cv.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=1)
            show_image(image)
    return


if __name__ == '__main__':
    # demo1()
    demo2()

"""
神经网络模型.
网络的部分单独写在这里, 因为在训练模型和加载已训练模型时都需要用到这一部分 ( Graph - 图).
原创作者的输入图片是 (None, 128, 128, 3), 但是我的电脑可能内存不够, 我就转成了 (None, 64, 64, 1) 灰度图.
"""
import tensorflow as tf


class Model(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1), name='x')
        self.y = tf.placeholder(tf.int32, shape=(None, 60), name='y')

    def neural_networks(self, x):
        # x.shape = (None, 64, 64, 1). filters 参数代表输出 32 个通道.  即: (None, 64, 64, 32)
        conv1 = tf.layers.conv2d(inputs=x,
                                 filters=32,
                                 kernel_size=(5, 5),
                                 padding='SAME',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        # (None, 64, 64, 32) -> (None, 32, 32, 32)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2)

        # (None, 32, 32, 32) -> (None, 32, 32, 64)
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=(5, 5),
                                 padding='SAME',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        # (None, 32, 32, 64) -> (None, 16, 16, 64)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2)

        # (None, 16, 16, 64) -> (None, 16, 16, 128)
        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=128,
                                 kernel_size=(3, 3),
                                 padding='SAME',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        # (None, 16, 16, 128) -> (None, 8, 8, 128)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2, 2), strides=2)
        # (None, 8, 8, 128) -> (None, 8*8*128) = (None, 8192)
        re1 = tf.reshape(pool3, shape=(-1, 8 * 8 * 128))

        # (None, 8192) -> (None, 1024)
        dense1 = tf.layers.dense(inputs=re1,
                                 units=1024,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        # (None, 1024) -> (None, 512)
        dense2 = tf.layers.dense(inputs=dense1,
                                 units=512,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        # # (None, 512) -> (None, 60)
        logits = tf.layers.dense(inputs=dense2,
                                 units=60,
                                 activation=None,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        return logits


model = Model()

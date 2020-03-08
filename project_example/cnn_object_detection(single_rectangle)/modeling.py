import tensorflow as tf


class Model(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1), name="x")
        self.y = tf.placeholder(tf.float32, shape=(None, 4), name="y")

    def neural_networks(self, x):
        # (None, 32, 32, 1) -> (None, 32, 32, 16)
        conv1 = tf.layers.conv2d(inputs=x,
                                 filters=16,
                                 kernel_size=(3, 3),
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 name="conv1")
        # (None, 32, 32, 16) -> (None, 16, 16, 16)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2, name="pool1")
        # (None, 16, 16, 16) -> (None, 16, 16, 32)
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=32,
                                 kernel_size=(3, 3),
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 name="conv2")
        # (None, 16, 16, 32) -> (None, 8, 8, 32)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2, name="pool2")
        # (None, 8, 8, 32) -> (None, 8, 8, 64)
        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=64,
                                 kernel_size=(3, 3),
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 name="conv3")
        # (None, 8, 8, 64) -> (None, 4, 4, 64)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2, 2), strides=2, name="pool3")
        # (None, 4, 4, 64) -> (None, 1024)
        reshape1 = tf.reshape(pool3, shape=(-1, 4 * 4 * 64), name="reshape1")
        # (None, 1024) -> (None, 256)
        dense1 = tf.layers.dense(inputs=reshape1,
                                 units=256,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                 name="dense1")

        # (None, 256) -> (None, 32)
        dense2 = tf.layers.dense(inputs=dense1,
                                 units=32,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                 name="dense2")

        # (None, 32) -> (None, 4)
        dense3 = tf.layers.dense(inputs=dense2,
                                 units=4,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                 name="dense3")

        return dense3


_model = None


def get_model():
    global _model
    if _model is None:
        _model = Model()
    return _model

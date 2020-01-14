import tensorflow as tf


class Model(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=(None, 128, 160, 3), name="x")
        self.y = tf.placeholder(tf.float32, shape=(None, 4), name="y")

    def neural_networks(self, x):
        # (None, 128, 160, 3) -> (None, 128, 160, 32)
        conv1 = tf.layers.conv2d(inputs=x,
                                 filters=32,
                                 kernel_size=(5, 5),
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 name="conv1")
        # (None, 128, 160, 32) -> (None, 64, 80, 32)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2, name="pool1")
        # (None, 64, 80, 32) -> (None, 64, 80, 64)
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=(5, 5),
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 name="conv2")
        # (None, 64, 80, 64) -> (None, 32, 40, 64)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=2, name="pool2")
        # (None, 32, 40, 64) -> (None, 32, 40, 128)
        conv3 = tf.layers.conv2d(inputs=pool2,
                                 filters=128,
                                 kernel_size=(3, 3),
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 name="conv3")
        # (None, 32, 40, 128) -> (None, 16, 20, 128)
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=(2, 2), strides=2, name="pool3")
        # (None, 16, 20, 128) -> (None, 16, 20, 256)
        conv4 = tf.layers.conv2d(inputs=pool3,
                                 filters=256,
                                 kernel_size=(3, 3),
                                 padding="SAME",
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 name="conv4")
        # (None, 16, 20, 256) -> (None, 8, 10, 256)
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=(2, 2), strides=2, name="pool4")
        # (None, 8, 10, 256) -> (None, 20480)
        reshape1 = tf.reshape(pool4, shape=(-1, 8 * 10 * 256), name="reshape1")
        # (None, 20480) -> (None, 5120)
        dense1 = tf.layers.dense(inputs=reshape1,
                                 units=5120,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                 name="dense1")

        # (None, 5120) -> (None, 1280)
        dense2 = tf.layers.dense(inputs=dense1,
                                 units=1280,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                 name="dense2")

        # (None, 1280) -> (None, 256)
        dense3 = tf.layers.dense(inputs=dense2,
                                 units=256,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                 name="dense3")
        # (None, 256) -> (None, 4)
        dense4 = tf.layers.dense(inputs=dense3,
                                 units=4,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003),
                                 name="dense4")
        return dense4


_model = None


def get_model():
    global _model
    if _model is None:
        _model = Model()
    return _model

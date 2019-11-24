"""
p_net 参考链接:
https://github.com/LeslieZhoa/tensorflow-MTCNN/blob/master/train/model.py

我现在还搞不清楚命名空间的具体用法呀.
我不想用他使用的那个 slim. 改成简单清晰一点的代码.

"""
import tensorflow as tf


x = tf.placeholder(tf.float32, shape=(None, 12, 12, 1), name='x')
y = tf.placeholder(tf.int32, shape=(None, 3), name='y')

# (None, 12, 12, 1) -> (None, 10, 10, 10)
conv1 = tf.layers.conv2d(inputs=x,
                         filters=10,
                         kernel_size=(3, 3),
                         padding='VALID',
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

# (None, 10, 10, 10) -> (None, 5, 5, 10)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=2)

# (None, 5, 5, 10) -> (None, 3, 3, 16)
conv2 = tf.layers.conv2d(inputs=pool1,
                         filters=16,
                         kernel_size=(3, 3),
                         padding='VALID',
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

# (None, 3, 3, 16) -> (None, 1, 1, 32)
conv3 = tf.layers.conv2d(inputs=conv2,
                         filters=32,
                         kernel_size=(3, 3),
                         padding='VALID',
                         activation=tf.nn.relu,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

# cls_prob (None, 1, 1, 32) -> (None, 1, 1, 2)
conv4_1 = tf.layers.conv2d(inputs=conv3,
                           filters=2,
                           kernel_size=(1, 1),
                           padding='VALID',
                           activation=tf.nn.softmax,
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
# bbox_pred (None, 1, 1, 32) -> (None, 1, 1, 4)
conv4_2 = tf.layers.conv2d(inputs=conv3,
                           filters=4,
                           kernel_size=(1, 1),
                           padding='VALID',
                           activation=None,
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
# landmark_pred (None, 1, 1, 32) -> (None, 1, 1, 10)
conv4_3 = tf.layers.conv2d(inputs=conv3,
                           filters=10,
                           kernel_size=(1, 1),
                           padding='VALID',
                           activation=None,
                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

# (None, 1, 1, 2) -> (None, 2)
cls_prob = tf.squeeze(conv4_1, [1, 2], name='cls_prob')

































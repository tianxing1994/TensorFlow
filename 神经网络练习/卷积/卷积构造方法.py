

import tensorflow as tf
import numpy as np

# 解耦函数

# tf.nn.relu
# n = tf.constant(np.array([1,2,-6,0,-3]))
# relu = tf.nn.relu(n)
# with tf.Session() as sess:
#     print(sess.run(relu))


# tf.nn.softplus
n = tf.constant(np.array([1,2,-6,0,-3], dtype=np.float32))
relu = tf.nn.softplus(n)
with tf.Session() as sess:
    print(sess.run(relu))

print(np.log(np.e + 1))











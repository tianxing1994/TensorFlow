"""
卷积 convolution


"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

moon = plt.imread(r"C:\Users\tianx\PycharmProjects\TensorFlow\datasets\moonlanding.png")
# plt.imshow(moon,cmap="gray")
# plt.show()

input = tf.constant(moon.reshape(1,474,630,1),dtype=tf.float32)

# 均值卷积盒
filter = tf.constant(np.full(shape=(3,3,1,1),fill_value=1/9), dtype=tf.float32)

# 高斯卷积盒
# filter = tf.constant(np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]]).reshape(3,3,1,1), dtype=tf.float32)


conv = tf.nn.conv2d(input=input,filter=filter, strides=[1,1,1,1], padding='SAME')

with tf.Session() as sess:
    moon_conv = sess.run(conv)
    # moon_conv.shape = (1, 474, 630, 1)

plt.imshow(moon_conv.reshape(474,630), cmap="gray")
plt.show()




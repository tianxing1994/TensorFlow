import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


cat = plt.imread(r"C:\Users\tianx\PycharmProjects\TensorFlow\datasets\cat.jpg")

cat = cat.mean(axis=2)

print(cat.shape)

input = tf.constant(cat.reshape(1,456,730,1),dtype=tf.float32)

filter = tf.constant(np.array([[-1, -1, 0], [-1, 0, 1], [0,1,1]]).reshape(3,3,1,1), dtype=tf.float32)

conv = tf.nn.conv2d(input=input,filter=filter, strides=[1,1,1,1], padding='SAME')

with tf.Session() as sess:
    cat_conv = sess.run(conv)

plt.imshow(cat_conv.reshape(456,730), cmap="gray")
plt.show()
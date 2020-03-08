import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


europe = plt.imread(r"C:\Users\tianx\PycharmProjects\TensorFlow\datasets\欧式.jpg")

# 将 RGB 的维度转置到 axis=0 维. 把原图当作三幅图来处理, 处理批次为 3.
input = tf.constant(europe.transpose([2,0,1]).reshape(3,582,1024,1), dtype=tf.float32)
filter = tf.constant(np.array([[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]]).reshape(3,3,1,1), dtype=tf.float32)
conv = tf.nn.conv2d(input=input,filter=filter, strides=[1,1,1,1], padding='SAME')

with tf.Session() as sess:
    europe_conv = sess.run(conv)
    # europe_conv.shape = (3, 582, 1024, 1)

# 将 RGB 取值范围变成[0，1]之间，因为图片信息的数值矩阵类型为 unit8 型，在 0~255 范围内，而在数据处理时使用 float32 型，当大于 1 时就会被显示成白色，不能有效表达图片信息。
plt.imshow(europe_conv.transpose([1,2,0,3]).reshape(582,1024,3) / 255)
plt.show()







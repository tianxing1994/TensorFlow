from __future__ import division, print_function, absolute_import
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


mnist = input_data.read_data_sets(r"C:\Users\tianx\PycharmProjects\TensorFlow\datasets\data", one_hot=True)

# mnist 数据集包含 55000 个训练样本, 每个样本为 784 的一维数组, 代表 28 * 28 的图片. 同样的有 10000 个训练数据样本.
# mnist.train.images.shape: (55000, 784)
# mnist.test.images.shape: (10000, 784)
# mnist.validation: <tensorflow.contrib.learn.python.learn.datasets.mnist.DataSet object at 0x0000021235B06898>
# print(dir(mnist.train)): 'epochs_completed', 'images', 'labels', 'next_batch', 'num_examples'.
# print(mnist.train.labels[0]): [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
# print(mnist.train.labels.shape): (55000, 10)
# mnist.train.next_batch(10) 将迭代地输入 (X,y) 训练样本数组


X = tf.placeholder(dtype=tf.float32, shape=(None,784))

w1 = tf.Variable(initial_value=tf.random_normal([784, 256]), dtype=tf.float32)
b1 = tf.Variable(initial_value=tf.random_normal([256]), dtype=tf.float32)

w2 = tf.Variable(initial_value=tf.random_normal([256, 128]), dtype=tf.float32)
b2 = tf.Variable(initial_value=tf.random_normal([128]), dtype=tf.float32)

w3 = tf.Variable(initial_value=tf.random_normal([128,256]), dtype=tf.float32)
b3 = tf.Variable(initial_value=tf.random_normal([256]), dtype=tf.float32)

w4 = tf.Variable(initial_value=tf.random_normal([256,784]), dtype=tf.float32)
b4 = tf.Variable(initial_value=tf.random_normal([784]), dtype=tf.float32)

def encode_op(X):
    X1 = tf.nn.sigmoid(tf.add(tf.matmul(X,w1), b1))
    X2 = tf.nn.sigmoid(tf.add(tf.matmul(X1,w2), b2))
    return X2


def decode_op(X2):
    X3 = tf.nn.sigmoid(tf.add(tf.matmul(X2,w3), b3))
    X4 = tf.nn.sigmoid(tf.add(tf.matmul(X3,w4), b4))
    return X4

encode_op = encode_op(X)
decode_op = decode_op(encode_op)
y_predictions = decode_op
y_true = X

# 求方差
loss = tf.reduce_mean(tf.pow(y_true - y_predictions, 2))

optimizer = tf.train.RMSPropOptimizer(0.001).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(1, 20000+1):
    batch_x, _ = mnist.train.next_batch(200)

    _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
    if i % 200 == 0 or i == 1:
        print('Step %i: Minibatch Loss: %f' % (i, l))

n = 4
canvas_orig = np.empty((28 * n, 28 * n))
canvas_recon = np.empty((28 * n, 28 * n))
for i in range(n):
    # MNIST test set
    batch_x, _ = mnist.test.next_batch(n)
    # Encode and decode the digit image
    g = sess.run(decode_op, feed_dict={X: batch_x})

    # Display original images
    for j in range(n):
        # Draw the generated digits
        canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = batch_x[j].reshape([28, 28])
    # Display reconstructed images
    for j in range(n):
        # Draw the generated digits
        canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

print("Original Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.show()

print("Reconstructed Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()









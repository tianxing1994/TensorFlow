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


w1 = tf.Variable(initial_value=tf.random_normal(shape=[784, 512], stddev=1./tf.sqrt(784/2)))
w_mean1 = tf.Variable(initial_value=tf.random_normal(shape=[512, 2], stddev=1./tf.sqrt(512/2)))
w_std1 = tf.Variable(initial_value=tf.random_normal(shape=[512, 2], stddev=1./tf.sqrt(512/2)))
w2 = tf.Variable(initial_value=tf.random_normal(shape=[2, 512], stddev=1./tf.sqrt(2/2)))
w3 = tf.Variable(initial_value=tf.random_normal(shape=[512, 784], stddev=1./tf.sqrt(512/2)))


b1 = tf.Variable(initial_value=tf.random_normal(shape=[512], stddev=1./tf.sqrt(512/2)))
b_mean1 = tf.Variable(initial_value=tf.random_normal(shape=[2], stddev=1./tf.sqrt(2/2)))
b_std1 = tf.Variable(initial_value=tf.random_normal(shape=[2], stddev=1./tf.sqrt(2/2)))
b2 = tf.Variable(initial_value=tf.random_normal(shape=[512], stddev=1./tf.sqrt(512/2)))
b3 = tf.Variable(initial_value=tf.random_normal(shape=[784], stddev=1./tf.sqrt(784/2)))

X = tf.placeholder(dtype=tf.float32, shape=(None,784))

X1 = tf.nn.tanh(tf.add(tf.matmul(X, w1), b1))

X_mean1 = tf.nn.tanh(tf.add(tf.matmul(X1, w_mean1), b_mean1))
X_std1 = tf.nn.tanh(tf.add(tf.matmul(X1, w_std1), b_std1))

X2 = X_mean1 + tf.exp(X_std1 / 2) * tf.random_normal(tf.shape(X_std1), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')


X3 = tf.nn.tanh(tf.add(tf.matmul(X2, w2), b2))
X4 = tf.nn.sigmoid(tf.add(tf.matmul(X3, w3), b3))

y_predictions = X4
y_true = X

information_entropy = - tf.reduce_sum(y_true * tf.log(1e-10 + y_predictions) + (1 - y_true) * tf.log(1e-10 + 1 - y_predictions), 1)

kl_div_loss = 1 + X_std1 - tf.square(X_mean1) - tf.exp(X_std1)
kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)

loss = tf.reduce_mean(information_entropy + kl_div_loss)
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(1, 20000+1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels)
    batch_x, _ = mnist.train.next_batch(200)

    # Train
    feed_dict = {X: batch_x}
    _, l = sess.run([optimizer, loss], feed_dict=feed_dict)
    if i % 1000 == 0 or i == 1:
        print('Step %i, Loss: %f' % (i, l))



noise_input = tf.placeholder(tf.float32, shape=[None, 2])
# Rebuild the decoder to create image from noise
decoder = tf.nn.tanh(tf.matmul(noise_input, w2) + b2)
decoder = tf.sigmoid(tf.matmul(decoder, w3) + b3)


# Building a manifold of generated digits
n = 20
x_axis = np.linspace(-3, 3, n)
y_axis = np.linspace(-3, 3, n)

canvas = np.empty((28 * n, 28 * n))
for i, yi in enumerate(x_axis):
    for j, xi in enumerate(y_axis):
        z_mu = np.array([[xi, yi]] * 200)
        x_mean = sess.run(decoder, feed_dict={noise_input: z_mu})
        canvas[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = \
        x_mean[0].reshape(28, 28)

plt.figure(figsize=(8, 10))
Xi, Yi = np.meshgrid(x_axis, y_axis)
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()







































































































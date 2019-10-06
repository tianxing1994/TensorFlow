from __future__ import print_function
import tensorflow as tf
import numpy as np
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
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


# <---------- 只用一层 ---------->
# w1 = tf.Variable(initial_value=tf.random_normal(shape=(784,10)),dtype=tf.float32)
# b1 = tf.Variable(initial_value=tf.random_normal(shape=(1,10)),dtype=tf.float32)
# y_1 = tf.matmul(X, w1) + b1
# y_prob = tf.math.softmax(y_1)
# <---------- 只用一层 ---------->

# <---------- 三层神经网络 ---------->
# 第一层神经元. 将 784 个特征转换成 128 个特征. 输出 y_1 形状为: (None, 128)
w1 = tf.Variable(initial_value=tf.random_normal(shape=(784,128)),dtype=tf.float32)
b1 = tf.Variable(initial_value=tf.random_normal(shape=(1,128)),dtype=tf.float32)
y_1 = tf.matmul(X, w1) + b1

# 第一层神经元. 将 128 个特征转换成 64 个特征. 输出 y_2 形状为: (None, 64)
w2 = tf.Variable(initial_value=tf.random_normal(shape=(128,64)),dtype=tf.float32)
b2 = tf.Variable(initial_value=tf.random_normal(shape=(1,64)),dtype=tf.float32)
y_2 = tf.matmul(y_1, w2) + b2


# 第三层神经元. 将 64 个特征转换为 (1,10) 的数组. 输出 y_ 形状为: (None, 10)
w3 = tf.Variable(initial_value=tf.random_normal(shape=(64,10)),dtype=tf.float32)
b3 = tf.Variable(initial_value=tf.random_normal(shape=(1,10)),dtype=tf.float32)
y_ = tf.matmul(y_2, w3) + b3

# 得出 y_ 是形状为: (None,10) 数组, 但其中的值不是 0-1 之间的.
# 其值被当作 y_ = log(probabilities), softmax() 就是返求解 probabilities
# 这里用 softmax 将其转换为概率, 即所有值都是 0-1 的数.
y_prob = tf.math.softmax(y_)
# <---------- 三层神经网络 ---------->

Y = tf.placeholder(shape=(None,10), dtype=tf.float32)

# 损失函数(求交叉熵的最小值) 1/n * ∑P * log(1/p)
# P 是预测出的概率, p 是真实的概率.
# p 中有为 0 的值, 为 0 的值会使 tf.log(p) 取无穷大. 因此要给它加上一个级小的数. 以防止出现 nan
cost = tf.reduce_mean(tf.reduce_sum(tf.multiply(Y, tf.log(1/(y_prob+1e-20))), axis=1))

# 指定优化器: AdamOptimizer
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        X_train, y_train = mnist.train.next_batch(1000)
        _,cost_ = sess.run([optimizer,cost], feed_dict={X:X_train,Y:y_train})
        print(cost_)

    else:
        # 计算准确率
        # 取出测试数据
        X_test, y_test = mnist.test.next_batch(1000)
        # 预测
        pred = sess.run(y_prob, {X:X_test})
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), tf.argmax(y_test, axis=1)), dtype=tf.float32))
        accuracy = sess.run(acc)
        print('当前准确率: %.4f' % (accuracy))

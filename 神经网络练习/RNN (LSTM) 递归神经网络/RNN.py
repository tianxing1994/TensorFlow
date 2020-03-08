from __future__ import print_function
import tensorflow as tf
from tensorflow import contrib
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

X = tf.placeholder(dtype=tf.float32, shape=(None,28,28))
Y = tf.placeholder(shape=(None,10), dtype=tf.float32)

# X2 为一个长度为 28 的列表, 其形状: [(None,28),(None,28) ... (None,28)]
X2 = tf.unstack(X, 28, axis=1)

# 创建长短期记忆单元. num_units=128 指定隐藏层的特征数为 128, 即其输出形状为: (None, 128)
lstm_cell = contrib.rnn.BasicLSTMCell(num_units=128, forget_bias=1.0)

# static_rnn() 的计算是: 将 28 个 (None,28) 输入 lstm_cell, 得出 28 个 (None, 128) 的结果.
# 其中, 每一次的输入每影响下一次输入得出的结果. 最后一次的结果是总结了前面所有输入而得出的结果.
# outputs 为一个长度为 28 的列表, 其形状: [(None,128),(None,128) ... (None,128)]
outputs, states = contrib.rnn.static_rnn(lstm_cell, X2, dtype=tf.float32)

w1 = tf.Variable(initial_value=tf.random_normal(shape=(128,10)))
b1 = tf.Variable(initial_value=tf.random_normal(shape=(1,10)))

# outputs[-1] 为最后一次输入得出的结果, 形状为 (None, 128).
logits = tf.matmul(outputs[-1], w1) + b1

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.math.softmax(logits),axis=1), tf.argmax(Y, axis=1)), tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(10000):
        batch_x, batch_y = mnist.train.next_batch(batch_size=128)
        batch_x = batch_x.reshape((128,28,28))
        sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y})

        if step % 200 == 0 or step == 1:
            cost_, accuracy_ = sess.run([cost, accuracy], feed_dict={X:batch_x, Y:batch_y})

            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(cost_) + ", Training Accuracy= " + \
                  "{:.3f}".format(accuracy_))

    else:
        test_data = mnist.test.images[:128].reshape((-1, 28, 28))
        test_label = mnist.test.labels[:128]
        print(sess.run(accuracy, feed_dict={X:test_data,Y:test_label}))

































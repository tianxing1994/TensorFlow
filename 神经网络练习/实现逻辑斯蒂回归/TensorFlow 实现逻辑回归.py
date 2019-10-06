# 预测函数 1 / (1 + power(e,-z))
# 损失函数: sum(y_test * tf.log(1/y_pred))
# 使用梯度下降法求损失函数最小的时候的系数, 此处的损失函数取交叉熵的最小值.
# 交叉熵: 真实概率乘以 log 以 2 为底的预测概率的倒数的值 P * np.log2(1/p).

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# 从数据集中读取数据.
mnist = input_data.read_data_sets(r'C:\Users\tianx\PycharmProjects\TensorFlow\datasets\data',one_hot=True)

# 训练样本: mnist.train.num_examples = 55000
# 测试样本: mnist.test.num_examples = 10000
# 样本图片: mnist.train.images.shape = (55000, 784)

# 独热编码: print(mnist.train.labels[0])
# 独热编码 one_hot: 如果要表示 0 到 9 的数, 则帛定一个列表, 其中只有一个值为 1,
# 其它全为 0, 每一个位置代表一个数据. 根据 1 出现的位置来表示值.
# 7 表示为: [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]


X = tf.placeholder(dtype=tf.float64, shape=(None, 784))
Y = tf.placeholder(dtype=tf.float64, shape=(None, 10))
W = tf.Variable(initial_value=np.zeros(shape=(784,10)), dtype=tf.float64)
B = tf.Variable(initial_value=np.zeros(shape=10), dtype=tf.float64)

# Y = W * X + B, tf.matmul() X, W 作矩阵点乘法, 再加 B, B 会自动广播至对应形状
y = tf.add(tf.matmul(X,W), B)

# softmax.
y_pred = tf.nn.softmax(y)

# 交叉熵: pk * log(1/ qk). 此处需取交叉熵最小时的 W, B.
cost = tf.reduce_mean(tf.reduce_sum(Y * tf.log(1/y_pred), axis=1))

init = tf.global_variables_initializer()

optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(init)
    for j in range(10):
        c_ = 0
        for i in range(500):
            # 分批传入数据流. 每次 100 条.
            X_train, y_train = mnist.train.next_batch(100)
            opt, c = sess.run([optimizer, cost], feed_dict={X:X_train, Y:y_train})

            c_ += c

        # 求每次 500 个数据之后的平均损失.
        loss = c_ / 550
        print(f"第{j}次大循环, 损失: {loss}")

    X_test, y_test = mnist.test.next_batch(5000)
    w = sess.run(W)
    b = sess.run(B)
    y_ = tf.nn.softmax(tf.add(tf.matmul(X_test.astype(np.float64), w), b))
    y_index = tf.argmax(y_, axis=1)
    y_test_index = tf.argmax(y_test, axis=1)
    result = tf.equal(y_index, y_test_index)

    accuracy = tf.reduce_mean(tf.cast(result, tf.float64))
    accu = sess.run(accuracy)
    print(f"准确率为: {accu}")

    # saver = tf.train.Saver()
    # saver.save(sess, "")




















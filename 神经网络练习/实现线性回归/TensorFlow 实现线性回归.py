"""https://github.com/aymericdamien/TensorFlow-Examples"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 创建数据. x, y 都是 0-10 之间等间距取 20 个数并加上偏差.
x = np.linspace(0, 10, 20) + np.random.rand(20) * 2
y = np.linspace(0, 10, 20) + np.random.rand(20) * 2

# y = ax + b -- Y = WX + B
X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)
W = tf.Variable(initial_value=np.random.randn(1), dtype=tf.float32)
B = tf.Variable(initial_value=np.random.randn(1), dtype=tf.float32)

# 构造线性方程
y_pred = tf.add(tf.multiply(X, W), B)

# 损失函数, 均方差
cost = tf.reduce_sum(tf.square(Y - y_pred)) / len(x)
# reduce_sum 等价于 numpy.sum()

# 梯度下降 - 使用梯度下降法找到让损失函数最小时的 W, B
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# 初始化变量
init = tf.global_variables_initializer()

epoch = 30000
with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        for X_trian, y_train in zip(x, y):
            sess.run(optimizer, feed_dict={X: X_trian, Y: y_train})

        # 每循环一百次, 打印.
        if i % 100 == 0:
            w = sess.run(W)
            b = sess.run(B)
            loss = sess.run(cost,feed_dict={X: x, Y: y})
            print(f"第{i}次训练, 系数是: {w}, 截距是: {b}, 损失是: {loss}")

    # 结束时打印
    w = sess.run(W)
    b = sess.run(B)
    print(f"训练结束: 系数是: {w}, 截距是: {b}")

X_plot = np.linspace(2, 10, 100)
y_tf = w * X_plot + b
plt.plot(X_plot, y_tf, c="g")
plt.scatter(x, y)
plt.show()

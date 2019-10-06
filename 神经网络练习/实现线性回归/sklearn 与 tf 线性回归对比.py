"""https://github.com/aymericdamien/TensorFlow-Examples"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.linear_model import LinearRegression

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = np.linspace(0,10,20) + np.random.rand(20) * 2
y = np.linspace(0,10,20) + np.random.rand(20) * 2
X_plot = np.linspace(2,10,100)

X = tf.placeholder(dtype=tf.float32)
Y = tf.placeholder(dtype=tf.float32)
W = tf.Variable(initial_value=np.random.randn(1),dtype=tf.float32)
B = tf.Variable(initial_value=np.random.randn(1),dtype=tf.float32)

y_pred = tf.add(tf.multiply(X,W),B)

cost = tf.reduce_sum(tf.square(Y-y_pred)) / len(x)

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()

epoch = 30000
with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        for X_trian, y_train in zip(x,y):
            sess.run(optimizer, feed_dict={X:X_trian, Y:y_train})

        if i % 100 == 0:
            w = sess.run(W)
            b = sess.run(B)
            loss = sess.run(cost,feed_dict={X:x, Y:y})
            print(f"第{i}次训练, 系数是: {w}, 截距是: {b}, 损失是: {loss}")

    w = sess.run(W)
    b = sess.run(B)
    print(f"训练结束: 系数是: {w}, 截距是: {b}")

y_tf = w * X_plot + b
plt.plot(X_plot, y_tf, c="g",linewidth=5)


linear = LinearRegression()
linear.fit(x.reshape(-1,1),y)
y_sklearn = linear.predict(X_plot.reshape(-1,1))

plt.plot(X_plot, y_sklearn, c="r")

plt.scatter(x,y)
plt.show()










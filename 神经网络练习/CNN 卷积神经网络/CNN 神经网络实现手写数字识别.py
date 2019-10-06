"""不收敛"""
import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets(r'C:\Users\tianx\PycharmProjects\TensorFlow\datasets\data',one_hot=True)
# mnist.train.images.shape = (55000, 784)


# 做卷积操作
# 定义 X 为数量 None 未知, 宽度为 784 的训练数据
X = tf.placeholder(dtype=tf.float64, shape=(None, 784))
# 卷积核的具体值, 需要由神经网络来计算, 所以我们将其定义为变量, 规定卷积核 5*5, 输出 32.
# 定义 filter 为长宽 5*5 的矩阵, 输入 1 个通道, 输出 32 个通道.
filter = tf.Variable(initial_value=tf.random.normal(shape=(5,5,1,32), dtype=tf.float64), dtype=tf.float64)

# 定义卷积核处理训练数据 X 的方式, ([批次, 高度 28, 宽度 28, 通道 1 个] 通道与卷积核的输入通道数一致).
# 输出形状为: (None, 28,28,32)
conv1 = tf.nn.conv2d(input=tf.reshape(X, shape=(-1,28,28,1)), filter=filter, strides=[1,1,1,1], padding='SAME')

# 给卷积处理后的结果加上偏差.
# 输出形状为: (None, 28,28,32)
conv1 = conv1 + tf.Variable(initial_value=np.zeros(shape=32, dtype=np.float64), dtype=tf.float64)

# 加上激活函数, tf.nn.relu() 大于 0 的数不变, 小于 0 的数置 0.
# 输出形状为: (None, 28,28,32)
conv1 = tf.nn.relu(conv1)

# 取最大值池化
# 输出形状为: (None, 14,14,32)
conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

# 重复卷积 - 偏差 - 激活 - 池化.
filter_2 = tf.Variable(initial_value=tf.random.normal(shape=(5,5,32,64), dtype=tf.float64), dtype=tf.float64)

conv2 = tf.nn.conv2d(input=conv1_pool, filter=filter_2,  strides=[1,1,1,1], padding='SAME')

conv2 = conv2 + tf.Variable(initial_value=np.zeros(shape=64, dtype=np.float64), dtype=tf.float64)

conv2 = tf.nn.relu(conv2)

# 输出形状为: (None, 14,14,64)
conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

# 全连接
# 把卷积之后的结果传给所有的神经元.
# 规定需要的神经元数量 1024.
conn = tf.reshape(conv2_pool, shape=(-1, 7*7*64))

w = tf.Variable(initial_value=tf.random_normal(shape=(7*7*64, 1024), dtype=tf.float64), dtype=tf.float64)
b = tf.Variable(initial_value=np.zeros(shape=1024, dtype=np.float64), dtype=tf.float64)

# 输出形状为: (None, 1024)
fully_connect = tf.matmul(conn, w) + b

# 输出形状为: (None, 1024)
result = tf.nn.relu(fully_connect)

# dropout
# 把 keep_prob 定义成占位符
keep_prob = tf.placeholder(dtype=tf.float64)
result_dropout = tf.nn.dropout(result, keep_prob=keep_prob)

# y = wx + b
linear_w = tf.Variable(initial_value=tf.random_normal(shape=(1024,10), dtype=tf.float64), dtype=tf.float64)
linear_b = tf.Variable(initial_value=tf.random_normal(shape=(1,10), dtype=tf.float64), dtype=tf.float64)

# 输出形状为: (None, 1024) · (1024, 10) + (10) = (None, 10)
y_ = tf.matmul(result, linear_w) + linear_b

# 传到 softmax 中, 得到概率.
# 输出形状为: (None, 10)
y_prob = tf.nn.softmax(y_)

# 损失函数(求交叉熵的最小值) 1/n * ∑P * log(1/p)
# P 是预测出的概率, p 是真实的概率.
Y = tf.placeholder(shape=(None,10), dtype=tf.float64)

# 形状: (None, 10) × (None, 10)
# 输出结果为: None
cost = tf.reduce_mean(tf.reduce_sum(tf.multiply(Y, tf.log(1/(y_prob+0.0000001))), axis=1))


# AdamOptimizer 方法, 找损失函数最小时的系数.
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)

init = tf.global_variables_initializer()

# 训练
with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        # 每次从数据中取一批 100 个数据
        X_train, y_train = mnist.train.next_batch(100)
        _, cost_ = sess.run([optimizer,cost], feed_dict={X:X_train,Y:y_train,keep_prob:0.5})
        # 每训练 10 次, 1000 个数据.

        if i % 100 == 0:
            print(f"第{i}次训练, 损失: {cost_}")

            # 计算准确率
            # 取出测试数据
            X_test, y_test = mnist.test.next_batch(1000)
            # 预测
            pred = sess.run(y_prob, {X:X_test, keep_prob:1})
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), tf.argmax(y_test, axis=1)), dtype=tf.float64))
            accuracy = sess.run(acc)
            print('当前准确率: %.4f' % (accuracy))


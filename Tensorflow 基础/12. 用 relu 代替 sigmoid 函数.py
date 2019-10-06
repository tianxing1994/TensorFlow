"""
用 relu 代替 sigmoid 函数后, 应将学习率由 0.001 改到 0.0001
!!! 不收敛.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time


x = tf.placeholder('float32', [None, 784])
y_ = tf.placeholder('float32', [None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# 卷积核, 大小 5×5, 输入通道 1 个, 输出通道 6 个.
filter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.1), name="filter1")
bias1 = tf.Variable(tf.truncated_normal((6,)), name="bias1")
conv1 = tf.nn.conv2d(x_image, filter1, strides=[1, 1, 1, 1], padding="SAME", name="conv1")
h_conv1 = tf.nn.relu(conv1 + bias1, name="h_conv1")

max_pool1 = tf.nn.max_pool(h_conv1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="max_pool2")

filter2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.1), name="filter2")
bias2 = tf.Variable(tf.truncated_normal((16,)), name="bias2")
conv2 = tf.nn.conv2d(max_pool1, filter2, strides=(1, 1, 1, 1), padding="SAME", name="conv2")
h_conv2 = tf.nn.relu(conv2 + bias2, name="h_conv2")

max_pool2 = tf.nn.max_pool(h_conv2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="max_pool3")

w_fc1 = tf.Variable(tf.truncated_normal((7*7*16, 1024), stddev=0.1), name="w_fc1")
b_fc1 = tf.Variable(tf.truncated_normal((1024,)), name="b_fc1")
h_pool2_flat = tf.reshape(max_pool2, (-1, 7*7*16), name="h_pool2_flat")

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1, name="h_fc1")

# w_fc2 = tf.Variable(tf.truncated_normal((1024, 128), stddev=0.1), name="w_fc2")
# b_fc2 = tf.Variable(tf.truncated_normal((128,)), name="b_fc2")
# h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2, name="h_fc2")

w_fc3 = tf.Variable(tf.truncated_normal((1024, 10), stddev=0.1), name="w_fc3")
b_fc3 = tf.Variable(tf.truncated_normal((10,)), name="b_fc3")
y_conv = tf.nn.softmax(tf.matmul(h_fc1, w_fc3) + b_fc3, name="y_conv")

# 损失函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv), name="cross_entropy")
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

sess = tf.InteractiveSession()

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

sess.run(tf.global_variables_initializer())

mnist_data_set = input_data.read_data_sets("MNIST_data", one_hot=True)

for i in range(20000):
    batch_xs, batch_ys = mnist_data_set.train.next_batch(200)

    if i % 100 == 0:
        cost, train_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
        print("cost:", cost)

        # train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    train_step.run(feed_dict={x: batch_xs, y_: batch_ys})

sess.close()
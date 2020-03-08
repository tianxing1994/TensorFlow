"""
相关函数:
tf.train.Saver
saver.save
saver.restore
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time


x = tf.placeholder('float', [None, 784])
y_ = tf.placeholder('float', [None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

# 卷积核, 大小 5×5, 输入通道 1 个, 输出通道 6 个.
filter1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6]), name="filter1")
bias1 = tf.Variable(tf.truncated_normal([6]), name="bias1")
conv1 = tf.nn.conv2d(x_image, filter1, strides=[1, 1, 1, 1], padding="SAME", name="conv1")
h_conv1 = tf.nn.sigmoid(conv1 + bias1, name="h_conv1")

max_pool2 = tf.nn.max_pool(h_conv1, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="max_pool2")

filter2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16]), name="filter2")
bias2 = tf.Variable(tf.truncated_normal([16]), name="bias2")
conv2 = tf.nn.conv2d(max_pool2, filter2, strides=(1, 1, 1, 1), padding="SAME", name="conv2")
h_conv2 = tf.nn.sigmoid(conv2 + bias2, name="h_conv2")

max_pool3 = tf.nn.max_pool(h_conv2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="max_pool3")

filter3 = tf.Variable(tf.truncated_normal([5, 5, 16, 120]), name="filter3")
bias3 = tf.Variable(tf.truncated_normal([120]), name="bias3")
conv3 = tf.nn.conv2d(max_pool3, filter3, strides=(1, 1, 1, 1), padding="SAME", name="conv3")
h_conv3 = tf.nn.sigmoid(conv3 + bias3, name="h_conv3")

w_fc1 = tf.Variable(tf.truncated_normal((7 * 7 * 120, 80)), name="w_fc1")
b_fc1 = tf.Variable(tf.truncated_normal([80]), name="b_fc1")
h_pool2_flat = tf.reshape(h_conv3, (-1, 7 * 7 * 120), name="h_pool2_flat")

h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, w_fc1) + b_fc1, name="h_fc1")

w_fc2 = tf.Variable(tf.truncated_normal((80, 10)), name="w_fc2")
b_fc2 = tf.Variable(tf.truncated_normal([10]), name="b_fc2")
y_conv = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2, name="y_conv")

# 损失函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv), name="cross_entropy")
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

sess = tf.InteractiveSession()

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.global_variables_initializer())

mnist_data_set = input_data.read_data_sets("MNIST_data", one_hot=True)

for i in range(20000):
    batch_xs, batch_ys = mnist_data_set.train.next_batch(200)

    # train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
    _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})

    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        print("step %d, training accuracy %g, loss %s" % (i, train_accuracy, loss))


# 保存模型 (保存会话)
saver = tf.train.Saver()
save_path = saver.save(sess,
                       "C:/Users/Administrator/PycharmProjects/TensorFlow/datasets/models/LeNet_model/model.ckpt")

sess.close()
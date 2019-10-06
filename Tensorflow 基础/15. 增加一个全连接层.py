"""
使用的是 relu 函数, 收敛. 之前不收敛, 可能是应为对于激活函数 relu, 其神经元个数不够.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    result = tf.nn.conv2d(x, w, strides=(1, 1, 1, 1), padding="SAME")
    return result


def max_pool_2x2(x):
    result = tf.nn.max_pool(x, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")
    return result


sess = tf.InteractiveSession()

x = tf.placeholder("float32", [None, 784])
y_ = tf.placeholder("float32", [None, 10])

x_image = tf.reshape(x, (-1, 28, 28, 1))

w_conv1 = weight_variable((5, 5, 1, 32))
b_conv1 = bias_variable((32,))
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weight_variable((5, 5, 32, 64))
b_conv2 = bias_variable((64,))
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

w_fc1 = weight_variable((7*7*64, 1024))
b_fc1 = bias_variable((1024,))
h_pool2_flat = tf.reshape(h_pool2, (-1, 7*7*64))
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

w_fc2 = weight_variable((1024, 128))
b_fc2 = bias_variable((128,))
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

w_fc3 = weight_variable((128, 10))
b_fc3 = bias_variable((10,))
y_conv = tf.nn.softmax(tf.matmul(h_fc2, w_fc3) + b_fc3)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))

train_step = tf.train.GradientDescentOptimizer(1e-5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

sess.run(tf.global_variables_initializer())

mnist_data_set = input_data.read_data_sets("MNIST_data", one_hot=True)

for i in range(1000):
    batch_xs, batch_ys = mnist_data_set.train.next_batch(200)
    _, loss = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})

    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        print("step %d, training accuracy %g, loss %s" % (i, train_accuracy, loss))

sess.close()


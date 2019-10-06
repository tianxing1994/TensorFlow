"""
百度百科:
GAN的基本原理其实非常简单，这里以生成图片为例进行说明。假设我们有两个网络，G（Generator）和D（Discriminator）。
正如它的名字所暗示的那样，它们的功能分别是：
G是一个生成图片的网络，它接收一个随机的噪声z，通过这个噪声生成图片，记做G(z)。
D是一个判别网络，判别一张图片是不是“真实的”。它的输入参数是x，x代表一张图片，输出D（x）代表x为真实图片的概率，如果为1，就代表100%是真实的图片，而输出为0，就代表不可能是真实的图片。

在训练过程中，生成网络G的目标就是尽量生成真实的图片去欺骗判别网络D。而D的目标就是尽量把G生成的图片和真实的图片分别开来。
这样，G和D构成了一个动态的“博弈过程”。
最后博弈的结果是什么？在最理想的状态下，G可以生成足以“以假乱真”的图片G(z)。对于D来说，它难以判定G生成的图片究竟是不是真实的，
因此D(G(z)) = 0.5。
这样我们的目的就达成了：我们得到了一个生成式的模型G，它可以用来生成图片。 [3]
"""
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
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


# Training Params
num_steps = 70000
batch_size = 128
learning_rate = 0.0002

# Network Params
image_dim = 784     # 28*28 pixels
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 100     # Noise data points


def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))


weights = {
    'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),
    'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim])),
    'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim])),
    'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1])),
}

biases = {
    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),
    'gen_out': tf.Variable(tf.zeros([image_dim])),
    'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),
    'disc_out': tf.Variable(tf.zeros([1])),
}


def generator(x):
    hidden_layer = tf.matmul(x, weights['gen_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)

    out_layer = tf.matmul(hidden_layer, weights['gen_out'])
    out_layer = tf.add(out_layer, biases['gen_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


def discriminator(x):
    hidden_layer = tf.matmul(x, weights['disc_hidden1'])
    hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])
    hidden_layer = tf.nn.relu(hidden_layer)

    out_layer = tf.matmul(hidden_layer, weights['disc_out'])
    out_layer = tf.add(out_layer, biases['disc_out'])
    out_layer = tf.nn.sigmoid(out_layer)
    return out_layer


gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')
disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')

gen_sample = generator(gen_input)

disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample)

# 图像生成器的工作是生成出逼真的图片, 使之通过 discriminator() 函数后的值 disc_fake 逼近于 1.
gen_loss = -tf.reduce_mean(tf.log(disc_fake))

# 图像鉴别器的工作是, 使 disc_fake 的值变小, 使 disc_real 的值变大. 即, 区别出真假图片.
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))

# 图像生成器与图像鉴别器交替优化, 据说, 最终, 生成器能生成出非常逼真的图片.
# 鉴别器的能力虽然也在不断提高, 但最终的鉴别真假概率只能是: 0.5
optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)

gen_vars = [weights['gen_hidden1'], weights['gen_out'],
             biases['gen_hidden1'], biases['gen_out']]

disc_vars = [weights['disc_hidden1'], weights['disc_out'],
             biases['disc_hidden1'], biases['disc_out']]

train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for i in range(1, num_steps + 1):
    batch_x, _ = mnist.train.next_batch(batch_size)
    z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

    feed_dict = {disc_input: batch_x, gen_input: z}
    _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss], feed_dict=feed_dict)

    if i % 2000 == 0 or i == 1:
        print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

n = 6
canvas = np.empty((28 * n, 28 * n))
for i in range(n):
    # Noise input.
    z = np.random.uniform(-1., 1., size=[n, noise_dim])
    # Generate image from noise.
    g = sess.run(gen_sample, feed_dict={gen_input: z})
    # Reverse colours for better display
    g = -1 * (g - 1)
    for j in range(n):
        # Draw the generated digits
        canvas[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = g[j].reshape([28, 28])

plt.figure(figsize=(n, n))
plt.imshow(canvas, origin="upper", cmap="gray")
plt.show()

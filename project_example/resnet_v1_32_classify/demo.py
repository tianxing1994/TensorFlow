# 添加环境变量.
import os
import sys
p = os.getcwd()
sys.path.append(p)


import tensorflow as tf

from resnet_v1_32 import resnet_v1_32
from cifar10 import batch_load_cifar10_test


def sparse_accuarcy(labels, logits):
    labels = tf.cast(labels, dtype=tf.int64)
    logits = tf.math.argmax(input=logits, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, logits), tf.float64))
    return accuracy


def demo1():
    x = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(dtype=tf.int32, shape=(None,))
    logits = resnet_v1_32(data=x, training=True)

    accuarcy = sparse_accuarcy(labels=y, logits=logits)

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=3)
    save_path = "model/resnet_v1_32/resnet_v1_32.ckpt-2800"
    saver.restore(sess=sess, save_path=save_path)

    for data, labels in batch_load_cifar10_test(batch_size=100, epoch=1):
        the_accuarcy = sess.run(accuarcy, feed_dict={x: data, y: labels})

        print(the_accuarcy)

    return


if __name__ == '__main__':
    demo1()

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


# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 128

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit


# Create the neural network
def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # (None, 784)
        x = x_dict['images']
        # (None, 28, 28, 1)
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        # (None, 28, 28, 32), padding='valid', filters=32, 输出 32 个通道.
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # (None, 14, 14, 32)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # (None, 26, 26, 64)
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # (None, 13, 13, 64)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # (None, 10816)
        fc1 = tf.contrib.layers.flatten(conv2)
        # (None, 1024)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        # (None, 1024)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        # # (None, 10)
        out = tf.layers.dense(fc1, 10)

    return out


# Define the model function (following TF Estimator Template)
def model_fn(features, labels, mode):
    # Build the neural network
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)

    # Predictions
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    # Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs


model = tf.estimator.Estimator(model_fn)


# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=num_steps)












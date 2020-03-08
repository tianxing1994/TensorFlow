import tensorflow as tf
import time
import matplotlib.pyplot as plt


with tf.device('/cpu:0'):
    learning_rate = 1e-4
    train_iters = 200
    batch_size = 50
    display_step = 5
    n_classes = 2
    n_fc1 = 4096
    n_fc2 = 2048

    x = tf.placeholder(tf.float32, (None, 227, 227, 3))
    y = tf.placeholder(tf.float32, (None, n_classes))

    W_conv = {
        'conv1': tf.Variable(tf.truncated_normal((11, 11, 3, 96), stddev=0.0001)),
        'conv2': tf.Variable(tf.truncated_normal((5, 5, 96, 256), stddev=0.01)),
        'conv3': tf.Variable(tf.truncated_normal((3, 3, 256, 384), stddev=0.01)),
        'conv4': tf.Variable(tf.truncated_normal((3, 3, 384, 384), stddev=0.01)),
        'conv5': tf.Variable(tf.truncated_normal((3, 3, 384, 256), stddev=0.01)),
        'fc1': tf.Variable(tf.truncated_normal((13 * 13 * 256, n_fc1), stddev=0.1)),
        'fc2': tf.Variable(tf.truncated_normal((n_fc1, n_fc2), stddev=0.1)),
        'fc3': tf.Variable(tf.truncated_normal((n_fc2, n_classes), stddev=0.1))
    }

    b_conv = {
        'conv1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=(96, ))),
        'conv2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=(256, ))),
        'conv3': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=(384, ))),
        'conv4': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=(384, ))),
        'conv5': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=(256, ))),
        'fc1': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=(n_fc1, ))),
        'fc2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=(n_fc2, ))),
        'fc3': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=(n_classes, )))
    }

    x_image = tf.reshape(x, (-1, 227, 227, 3))

    conv1 = tf.nn.conv2d(x_image, W_conv['conv1'], strides=(1, 4, 4, 1), padding='VALID')
    conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])
    conv1 = tf.nn.relu(conv1)

    pool1 = tf.nn.avg_pool(conv1, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')

    norm1 = tf.nn.lrn(pool1, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    conv2 = tf.nn.conv2d(norm1, W_conv['conv2'], strides=(1, 1, 1, 1), padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
    conv2 = tf.nn.relu(conv2)

    pool2 = tf.nn.avg_pool(conv2, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')

    norm2 = tf.nn.lrn(pool2, 5, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    conv3 = tf.nn.conv2d(norm2, W_conv['conv3'], strides=(1, 1, 1, 1), padding='SAME')
    conv3 = tf.nn.bias_add(conv3, b_conv['conv3'])
    conv3 = tf.nn.relu(conv3)

    conv4 = tf.nn.conv2d(conv3, W_conv['conv4'], strides=(1, 1, 1, 1), padding='SAME')
    conv4 = tf.nn.bias_add(conv4, b_conv['conv4'])
    conv4 = tf.nn.relu(conv4)

    conv5 = tf.nn.conv2d(conv4, W_conv['conv5'], strides=(1, 1, 1, 1), padding='SAME')
    conv5 = tf.nn.bias_add(conv5, b_conv['conv5'])
    conv5 = tf.nn.relu(conv5)

    pool5 = tf.nn.avg_pool(conv5, ksize=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding='VALID')

    reshape = tf.reshape(pool5, (-1, 13 * 13 * 256))

    fc1 = tf.add(tf.matmul(reshape, W_conv['fc1'], b_conv['fc1']))
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, 0.5)

    fc2 = tf.add(tf.matmul(fc1, W_conv['fc2']), b_conv['fc2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, 0.5)

    fc3 = tf.add(tf.matmul(fc2, W_conv['fc3']), b_conv['fc3'])

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc3, y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    correct_pred = tf.equal(tf.argmax(fc3, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


save_model = './model/AlexNetModel.ckpt'


def train(opech):
    with tf.Session() as sess:
        sess.run(init)

        train_writer = tf.summary.FileWriter('./log', sess.graph)
        saver = tf.train.Saver()

        c = list()
        start_time = time.time()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        step = 0

        for i in range(opech):
            step = i
            image, label = sess.run([image_batch, label_batch])
            labels = onehot(label)

            sess.run(optimizer, feed_dict={x: image, y: labels})
            loss_record = sess.run(loss, feed_dict={x: image, y: labels})
            print('now the loss is %f' % (loss_record, ))

            c.append(loss_record)
            end_time = time.time()

            print('time: ', (end_time - start_time))
            start_time = end_time
            print('------------------ %d onpech is finished ----------------------' % (i, ))

        print('Optimization Finished!')
        saver.save(sess, save_model)
        print('Model Save Finished!')

        coord.request_stop()
        coord.join(threads)

        plt.plot(c)
        plt.xlabel('Iter')
        plt.ylabel('loss')
        plt.tight_layout()
        plt.savefig('cat_and_dog_AlexNet.jpg', dpi=200)







import tensorflow as tf

from classical_net.vgg.vgg16.vgg16_flower_classify.load_data import batch_load_data
from classical_net.vgg.vgg16.vgg16_flower_classify.net import inference


x = tf.placeholder(tf.float32, [None, 224, 224, 3], name='x')
y = tf.placeholder(tf.int32, [None, 5], name='y')
y_ = inference(x)


def demo1():
    # 训练
    ckpt_path = '../../../../datasets/classical_model/vgg16/vgg_16.ckpt'

    restore_var_list = [var for var in tf.trainable_variables() if var.name.startswith("vgg_16")]
    optimize_var_list1 = [var for var in tf.trainable_variables() if var.name.startswith("vgg_16/conv5")]
    optimize_var_list2 = [var for var in tf.trainable_variables() if var.name.startswith("migration")]
    optimize_var_list2.extend(optimize_var_list1)

    correct_prediction = tf.equal(tf.cast(tf.argmax(y_, axis=1), tf.int32), tf.cast(tf.argmax(y, axis=1), tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_)

    saver1 = tf.train.Saver(var_list=restore_var_list, max_to_keep=3)
    saver2 = tf.train.Saver(max_to_keep=3)

    learning_rate = 1e-4
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=optimize_var_list2)

    n_epoch = 100
    batch_size = 32

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver1.restore(sess, ckpt_path)

        for epoch in range(n_epoch):
            train_loss, train_accuracy, n_batch = 0, 0, 0

            for x_train, y_train in batch_load_data(batch_size=batch_size, flag="train"):
                theloss, theAccuracy, _ = sess.run([loss, accuracy, train_op], feed_dict={x: x_train, y: y_train})
                train_loss += theloss
                train_accuracy += theAccuracy
                print(f"temp: theloss {theloss}, theAccuracy {theAccuracy}")
                n_batch += 1
            print(f"train loss: {train_loss / n_batch}, train accuracy: {train_accuracy / n_batch}")
            saver2.save(sess, "model/box_regression.ckpt", global_step=epoch + 1)
    return


if __name__ == '__main__':
    demo1()

"""
模型优化器
对神经网络模型 (model) 进行训练优化, 训练结果保存到 model 文件中.
"""
import tensorflow as tf

from project_example.face_recognition.load_data import DataLoader


class Optimizer(object):
    def __init__(self, data_loader, model, n_epoch=10, batch_size=64, learning_rate=0.001, show_validation=True):
        self._data_loader = data_loader
        self._model = model

        self._n_epoch = n_epoch
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._show_validation = show_validation

        self._x = self._model.x
        self._y = self._model.y
        self._y_ = tf.cast(tf.argmax(self._y, axis=1), tf.int32)

        self._logits = self._model.neural_networks(self._x)
        self._loss = tf.losses.sparse_softmax_cross_entropy(labels=self._y_, logits=self._logits)
        self._train_op = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._loss)

        correct_prediction = tf.equal(tf.cast(tf.argmax(self._logits, axis=1), tf.int32), self._y_)
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self._sess = tf.InteractiveSession()

    @staticmethod
    def _logger(string):
        with open('model/accuracy.txt', 'a+') as f:
            f.write(string)

    def _validation(self):
        validation_loss, validation_accuracy, n_batch = 0, 0, 0
        for x_val_a, y_val_a in self._data_loader.batch_load_data(data=self._data_loader.test_data,
                                                                  batch_size=self._batch_size,
                                                                  shuffle=False):
            err, ac = self._sess.run([self._loss, self._accuracy], feed_dict={self._x: x_val_a, self._y: y_val_a})
            validation_loss += err
            validation_accuracy += ac
            n_batch += 1
        print('validation loss: %f' % (validation_loss / n_batch))
        print('validation accuracy: %f' % (validation_accuracy / n_batch))
        return validation_loss, validation_accuracy

    def run(self, ckpt_path=None):
        saver = tf.train.Saver(max_to_keep=3)

        if ckpt_path is not None:
            saver.restore(self._sess, ckpt_path)
            print("Model restored from file: %s" % (ckpt_path,))

        self._sess.run(tf.global_variables_initializer())
        for epoch in range(self._n_epoch):
            train_loss, train_accuracy, n_batch = 0, 0, 0
            for x_train_a, y_train_a in self._data_loader.batch_load_data(data=self._data_loader.training_data,
                                                                          batch_size=self._batch_size,
                                                                          shuffle=False):
                # sess.run, 计算 train_op, loss, acc. 其中 train_op 是优化的必须操作, 其它的只为获取其值.
                _, loss, accuracy = self._sess.run(fetches=[self._train_op, self._loss, self._accuracy],
                                                   feed_dict={self._x: x_train_a, self._y: y_train_a})
                train_loss += loss
                train_accuracy += accuracy
                n_batch += 1
            print('train loss: %f' % (train_loss / n_batch))
            print('train accuracy: %f' % (train_accuracy / n_batch))

            if self._show_validation:
                validation_loss, validation_accuracy = self._validation()
                self._logger(str(epoch + 1) + ', validation_accuracy: ' + str(validation_accuracy) + '\n')

            saver.save(self._sess, 'model/faces.ckpt', global_step=epoch + 1)
        return


def demo1():
    from project_example.face_recognition.modeling import model

    data_path = 'dataset/faces'
    data_loader = DataLoader(data_path, grayscale=True, test_size=0.2, one_hot=True)
    optimizer = Optimizer(data_loader=data_loader, model=model, n_epoch=10, batch_size=64, learning_rate=0.001)
    optimizer.run()
    return


if __name__ == '__main__':
    demo1()

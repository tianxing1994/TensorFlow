import tensorflow as tf


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
        self._y_pred = self._model.neural_networks(self._x)

        self._loss = tf.reduce_sum(tf.square(self._y - self._y_pred)) / tf.cast(tf.shape(self._x)[0], tf.float32)
        self._train_op = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self._loss)
        self._sess = tf.InteractiveSession()

    @staticmethod
    def _logger(string):
        with open('model/accuracy.txt', 'a+') as f:
            f.write(string)

    def run(self, ckpt_path=None):
        saver = tf.train.Saver(max_to_keep=3)
        if ckpt_path is not None:
            saver.restore(self._sess, ckpt_path)
            print(f"Model restored from file: {ckpt_path}")

        self._sess.run(tf.global_variables_initializer())
        for epoch in range(self._n_epoch):
            train_loss, n_batch = 0, 0
            for i in range(100):
                x_train, y_train = self._data_loader(n_images=self._batch_size,
                                                     image_size=32,
                                                     min_object_size=2,
                                                     max_object_size=16)

                loss, _ = self._sess.run(fetches=[self._loss, self._train_op], feed_dict={self._x: x_train, self._y: y_train})

                train_loss += loss
                n_batch += 1
            print(f"train loss: {train_loss / n_batch}")

            saver.save(self._sess, "model/box_regression.ckpt", global_step=epoch + 1)

        return


def demo1():
    from load_data import generate_training_set
    from modeling import get_model
    model = get_model()
    optimizer = Optimizer(data_loader=generate_training_set, model=model, n_epoch=1000, batch_size=64, learning_rate=0.001)
    optimizer.run()
    return


if __name__ == '__main__':
    demo1()


















"""
加载训练好的模型 (ckpt 文件) 并对图片进行分类. 查看效果.
"""
import numpy as np
import tensorflow as tf


class Classifier(object):
    def __init__(self, model, ckpt_path):
        self._model = model
        self._ckpt_path = ckpt_path

        self._x = self._model.x
        self._y = self._model.y

        self._logits = self._model.neural_networks(self._x)
        self._classify = tf.argmax(self._logits, axis=1)

    def classify(self, gray_image):
        saver = tf.train.Saver(max_to_keep=3)
        image_data = np.expand_dims(gray_image, axis=0)
        image_data = np.expand_dims(image_data, axis=3)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, self._ckpt_path)
            print("Model restored from file: %s" % self._ckpt_path)
            result = sess.run([self._classify], feed_dict={self._x: image_data})
        return result


def demo1():
    import cv2 as cv
    from project_example.face_recognition.load_data import DataLoader
    from project_example.face_recognition.modeling import model

    classifier = Classifier(model=model, ckpt_path='model/faces.ckpt-1')

    data_path = 'dataset/faces'
    data_loader = DataLoader(data_path, grayscale=True, test_size=0.2, one_hot=True)
    data = data_loader.get_n_test_images(k=3, grayscale=True, one_hot=True)

    for image, target in data:
        result = classifier.classify(image)
        print(result)
        print(target)
    return


def demo2():
    import cv2 as cv
    from project_example.face_recognition.load_data import DataLoader
    from project_example.face_recognition.modeling import model

    classifier = Classifier(model=model, ckpt_path='model/faces.ckpt-1')

    face_path = 'D:/Users/Administrator/PycharmProjects/TensorFlow/project_example/face_recognition/dataset/faces/1511282/1001.png'
    image = cv.imread(face_path)
    image = cv.resize(image, dsize=(64, 64))
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    result = classifier.classify(gray_image)
    print(result)
    return


if __name__ == '__main__':
    # demo1()
    demo2()

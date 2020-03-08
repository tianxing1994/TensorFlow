"""
用于加载图片的模块.
由于一下子加载上万张图片, 电脑的内存不够用.
所以先将所有图片的路径及其对应的标签加载到内存中.
当需要输入模型时再读取图片.
"""
import os
import random
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


class DataLoader(object):
    def __init__(self, data_path, grayscale=True, test_size=0.2, one_hot=True):
        self._data_path = data_path
        self._grayscale = grayscale
        self._test_size = test_size
        self._one_hot = one_hot
        self._data = None
        self._training_data = None
        self._test_data = None
        self._labels = None

        self._one_hot_dict = dict()

    def _generate_one_hot(self, label):
        if label in self._one_hot_dict:
            return self._one_hot_dict[label]
        ll = len(self._labels)
        index = self._labels.index(label)

        one_hot = np.zeros(shape=(ll,), dtype=np.float)
        one_hot[index] = 1.
        self._one_hot_dict[label] = one_hot
        return self._one_hot_dict[label]

    @staticmethod
    def _load_image_path_and_labels(data_path):
        """
        读取所有图片的路径及其对应的标签. 打乱输出.
        :param data_path:
        :return: [(image_path, label), (image_path, label), ..., (image_path, label)]
        """
        data = list()
        labels = os.listdir(data_path)
        for label in labels:
            image_folder = os.path.join(data_path, label)
            image_names = os.listdir(image_folder)
            for image_name in image_names:
                image_path = os.path.join(image_folder, image_name)
                data.append((image_path, label))

        random.shuffle(data)
        return data, labels

    @property
    def data(self):
        if self._data is not None:
            return self._data
        self._data, self._labels = self._load_image_path_and_labels(self._data_path)
        return self._data

    @property
    def labels(self):
        if self._labels is not None:
            return self._labels
        self._data, self._labels = self._load_image_path_and_labels(self._data_path)
        return self._labels

    @staticmethod
    def _split_data(data, test_size=0.2):
        """
        将数据分割为训练集和测试集.
        :param data: [(image_path, label), (image_path, label), ..., (image_path, label)]
        :param test_size: 分割数据后, 测试数据所占的比例.
        :return:
        """
        l = len(data)
        s = int(l * test_size)
        train_data = data[s:]
        test_data = data[:s]
        return train_data, test_data

    @property
    def training_data(self):
        if self._training_data is not None:
            return self._training_data
        self._training_data, self._test_data = self._split_data(self.data, test_size=self._test_size)
        return self._training_data

    @property
    def test_data(self):
        if self._test_data is not None:
            return self._test_data
        self._training_data, self._test_data = self._split_data(self.data, test_size=self._test_size)
        return self._test_data

    def batch_load_data(self, data, batch_size=30, shuffle=False):
        """
        加载数据
        由于一下子加载几万张图片占用的内存很大, 所以, 只好先只加载其路径, 当需要时, 再分批读取图片.
        :param data: [(image_path, label), (image_path, label), ..., (image_path, label)]
        :param batch_size: 读取多少张图片.
        :param shuffle:
        :return:
        """
        if shuffle:
            np.random.shuffle(data)

        start_idx = 0
        inputs = list()
        targets = list()

        while True:
            batch_data = data[start_idx: start_idx+batch_size]
            for image_path, label in batch_data:
                image = cv.imread(image_path)
                image = cv.resize(image, dsize=(64, 64))
                if self._grayscale:
                    image = np.expand_dims(cv.cvtColor(image, cv.COLOR_BGR2GRAY), axis=2)

                inputs.append(image)
                if self._one_hot:
                    targets.append(self._generate_one_hot(label))
                else:
                    targets.append(label)
            start_idx += batch_size
            if start_idx >= len(data):
                break

            inputs_ = np.array(inputs, dtype=np.float64)
            targets_ = np.array(targets, dtype=np.float64)
            inputs = list()
            targets = list()
            yield inputs_, targets_

    def get_n_test_images(self, k, grayscale=True, one_hot=True):
        """
        任意加载几张测试图片
        :return:
        """
        random.shuffle(self.test_data)
        batch_data = self.test_data[:k]

        result = list()
        for image_path, label in batch_data:
            image = cv.imread(image_path)
            image = cv.resize(image, dsize=(64, 64))
            if grayscale:
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            if one_hot:
                target = np.argmax(self._generate_one_hot(label))
            else:
                target = label
            result.append((image, target))
        return result


def demo1():
    data_path = r'D:\Users\Administrator\PycharmProjects\TensorFlow\project_example\face_recognition\dataset\faces'
    data_loader = DataLoader(data_path, grayscale=True, test_size=0.2, one_hot=True)
    for x, y in data_loader.batch_load_data(data=data_loader.training_data, batch_size=32, shuffle=False):
        print(x.shape)
        print(type(x))
        print(y.shape)
        print(type(y))
    return


if __name__ == '__main__':
    demo1()

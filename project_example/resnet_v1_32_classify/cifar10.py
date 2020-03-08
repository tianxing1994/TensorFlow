import os
import pickle
import numpy as np


def cifar10_train():
    cifar10_dir = "dataset/cifar-10-batches-py"
    data = list()
    labels = list()
    for filename in os.listdir(cifar10_dir):
        if filename.startswith("data_batch_"):
            data_path = os.path.join(cifar10_dir, filename)

            with open(data_path, 'rb') as f:
                cifar_dict = pickle.load(f, encoding='bytes')

                _data = cifar_dict[b'data']
                _data = np.reshape(_data, newshape=(10000, 3, 32, 32))
                _labels = cifar_dict[b'labels']
                data.append(_data)
                labels.append(_labels)

    data = np.concatenate(data, axis=0)
    data = np.transpose(data, axes=[0, 2, 3, 1])
    labels = np.concatenate(labels, axis=0)
    return data, labels


def cifar10_test():
    test_batch = "dataset/cifar-10-batches-py/test_batch"
    with open(test_batch, 'rb') as f:
        cifar_dict = pickle.load(f, encoding='bytes')

        _data = cifar_dict[b'data']
        _data = np.reshape(_data, newshape=(10000, 3, 32, 32))
        _labels = cifar_dict[b'labels']
    data = np.transpose(_data, axes=[0, 2, 3, 1])
    labels = _labels
    return data, labels


def batch_load(data, labels, batch_size, epoch):
    n = data.shape[0]
    n_batch = n // batch_size
    index = np.arange(n)

    for i in range(epoch):
        np.random.shuffle(index)
        begin = 0
        end = batch_size
        for j in range(n_batch):
            yield data[begin: end], labels[begin: end]
            begin += batch_size
            end += batch_size


def batch_load_cifar10_train(batch_size=100, epoch=100):
    d, l = cifar10_train()
    for data, labels in batch_load(d, l, batch_size, epoch):
        yield data, labels


def batch_load_cifar10_test(batch_size=100, epoch=1):
    d, l = cifar10_test()
    for data, labels in batch_load(d, l, batch_size, epoch):
        yield data, labels


if __name__ == '__main__':
    cifar10_test()


import os
import cv2 as cv
import numpy as np


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    # cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def path_to_images(path_list, dsize=(224, 224)):
    ret = list()
    for path in path_list:
        image = cv.imread(str(path, 'utf-8'))
        image = cv.resize(image, dsize=dsize)
        ret.append(image)

    ret = np.stack(ret, axis=0)
    return ret


def label_to_onehot(label_list):
    daisy = [1, 0, 0, 0, 0]
    dandelion = [0, 1, 0, 0, 0]
    rose = [0, 0, 1, 0, 0]
    sunflower = [0, 0, 0, 1, 0]
    tulip = [0, 0, 0, 0, 1]
    ret = list()
    for label in label_list:
        label = str(label, 'utf-8')
        if label == 'daisy':
            ret.append(daisy)
        elif label == 'dandelion':
            ret.append(dandelion)
        elif label == 'rose':
            ret.append(rose)
        elif label == 'sunflower':
            ret.append(sunflower)
        elif label == 'tulip':
            ret.append(tulip)
        else:
            pass
    ret = np.array(ret, dtype=np.int32)
    return ret


def load_data_sheet(flower_dir="../../../../datasets/data/flowers"):
    data_list = list()
    label_list = list()
    for folder in os.listdir(flower_dir):
        label = folder
        class_dir = os.path.join(flower_dir, folder)
        for filename in os.listdir(class_dir):
            flower_path = os.path.join(class_dir, filename)
            data_list.append(flower_path)
            label_list.append(label)
    data = np.array(data_list, dtype=np.string_)
    label = np.array(label_list, dtype=np.string_)
    return data, label


def split_data(x, y, test_size=0.0):
    n = len(x)
    s = int(test_size * n)
    i = list(range(n))
    np.random.shuffle(i)
    x_train = x[i][s:]
    x_test = x[i][:s]
    y_train = y[i][s:]
    y_test = y[i][:s]
    return x_train, y_train, x_test, y_test


def load_data():
    data, target = load_data_sheet()
    x_train, y_train, x_test, y_test = split_data(data, target)
    return x_train, y_train, x_test, y_test


def _batch_load_data(x, y, batch_size=30):
    n = len(x)
    start_idx = 0
    while True:
        if start_idx >= n:
            break
        end_idx = start_idx + batch_size
        if end_idx > n:
            break

        x_batch = x[start_idx: end_idx]
        y_batch = y[start_idx: end_idx]

        x_batch = path_to_images(x_batch)
        y_batch = label_to_onehot(y_batch)
        start_idx += batch_size
        yield x_batch, y_batch


def batch_load_data(batch_size, flag="train"):
    if flag == "train":
        x_train, y_train, x_test, y_test = load_data()
        return _batch_load_data(x_train, y_train, batch_size=batch_size)
    else:
        x_train, y_train, x_test, y_test = load_data()
        return _batch_load_data(x_test, y_test, batch_size=batch_size)


def demo2():
    """批次加载数据. """
    for x, y in batch_load_data(batch_size=32, flag="train"):
        print(x)
        print(y)
        pass
    for x, y in batch_load_data(batch_size=32, flag="test"):
        print(x)
        print(y)
        pass
    return


if __name__ == '__main__':
    demo2()


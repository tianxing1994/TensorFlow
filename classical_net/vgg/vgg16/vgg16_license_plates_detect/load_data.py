import os
import cv2 as cv
import numpy as np
import pandas as pd


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    # cv.namedWindow(win_name, cv.WINDOW_AUTOSIZE)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def _load_data():
    # 数据集中的图片平均宽度为 657, 平均高度为 460. 这里我们将所有的图片 resize 到: 宽 640, 高 512.
    input_h, input_w = 224, 224
    dataset_dir = '../../../../datasets/data/license_plate'
    license_plates = pd.read_csv(os.path.join(dataset_dir, 'indian_license_plates.csv'))
    n, _ = license_plates.shape

    data = np.zeros(shape=(n, input_h, input_w, 3), dtype=np.float64)
    target = np.zeros(shape=(n, 4), dtype=np.float64)
    for index, row in license_plates.iterrows():
        image_name = row["image_name"]
        image_width = row["image_width"]
        image_height = row["image_height"]
        x1 = row["top_x"]
        y1 = row["top_y"]
        x2 = row["bottom_x"]
        y2 = row["bottom_y"]

        image_path = os.path.join(dataset_dir, 'indian_number_plates', image_name + ".jpeg")
        image = cv.imread(image_path)
        image_resized = cv.resize(image, dsize=(input_h, input_w))
        image_resized = np.array(image_resized, dtype=np.float32)
        data[index] = image_resized
        target[index] = [x1, y1, x2, y2]

    # shuffle_index = np.arange(len(data))
    # np.random.shuffle(shuffle_index)
    # data = data[shuffle_index]
    # target = target[shuffle_index]
    # print(data)
    # print(target)
    # print(data.shape)
    # print(target.shape)
    return data, target


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
    data, target = _load_data()
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
        start_idx += batch_size
        yield x_batch, y_batch


def batch_load_data(batch_size, flag="train"):
    if flag == "train":
        x_train, y_train, x_test, y_test = load_data()
        return _batch_load_data(x_train, y_train, batch_size=batch_size)
    else:
        x_train, y_train, x_test, y_test = load_data()
        return _batch_load_data(x_test, y_test, batch_size=batch_size)


def demo1():
    x_train, y_train, x_test, y_test = load_data()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    return


def demo2():
    """批次加载数据. """
    for x, y in batch_load_data(batch_size=32, flag="train"):
        print(x.shape)
        print(y.shape)
    for x, y in batch_load_data(batch_size=32, flag="test"):
        print(x.shape)
        print(y.shape)
    return


if __name__ == '__main__':
    # demo1()
    demo2()

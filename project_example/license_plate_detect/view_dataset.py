import os

import cv2 as cv
import numpy as np
import pandas as pd


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    """
    237 个车牌样本, 有 image_name, image_width, image_height, top_x, top_y, bottom_x, bottom_y 7 个字段.
    :return:
    """
    license_plates = pd.read_csv("dataset/indian_license_plates.csv")
    print(license_plates)
    return


def demo2():
    """
    遍历数据集的每一个样本, 并读取各字段值.
    :return:
    """
    license_plates = pd.read_csv("dataset/indian_license_plates.csv")

    print(license_plates.shape)
    print(license_plates.describe().loc["mean"])

    for index, row in license_plates.iterrows():
        print(index)
        print(row)
        print(type(row))
        print(row["image_name"])
        print(row["image_width"])
        print(row["image_height"])
        print(row["top_x"])
        print(row["top_y"])
        print(row["bottom_x"])
        print(row["bottom_y"])
        break
    return


def demo3():
    license_plates = pd.read_csv("dataset/indian_license_plates.csv")

    for index, row in license_plates.iterrows():
        image_name = row["image_name"]
        image_width = row["image_width"]
        image_height = row["image_height"]

        x1 = np.round(image_width * row["top_x"], decimals=0).astype("int")
        y1 = np.round(image_height * row["top_y"], decimals=0).astype("int")
        x2 = np.round(image_width * row["bottom_x"], decimals=0).astype("int")
        y2 = np.round(image_height * row["bottom_y"], decimals=0).astype("int")

        image_path = os.path.join("dataset/indian_number_plates", image_name + ".jpeg")
        image = cv.imread(image_path)
        image = cv.rectangle(image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=2)
        show_image(image)
    return


if __name__ == '__main__':
    demo2()

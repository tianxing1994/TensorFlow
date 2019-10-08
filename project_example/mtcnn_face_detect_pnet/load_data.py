"""
生成数据和加载数据.
我们的数据集中包含的是图片和其中人脸位置的 bounding box 标注. PNet 要求输入数据是 (None, 12, 12, 3).
也就是需要将原图像切割成 12*12 的大小 ROI 子图, 根据 bounding box 标注, 计算这个子图中是否包含人脸.
做一个分类训练, 分类结果为: pos, neg, part. 三种, 即: 全部为人脸部分, 没有人脸部分, 部分为人脸部分.

参考链接:
https://github.com/LeslieZhoa/tensorflow-MTCNN/tree/master/preprocess
"""
import os
import cv2 as cv
import numpy as np

# 数据集分类 lfw_5590, net_7876 两部分数据来源, 数据下载下来时, 已经有了 testImageList.txt, trainImageList.txt, 两个文件
# txt 文件中, 每行, 存放着图片地址, 人脸位置, 人脸 5 点特征的坐标. (每张图片中只存在一个人脸).
train_file_path = 'project_example/mtcnn_face_detect_pnet/dataset/train/trainImageList.txt'


class DataLoader(object):
    def __init__(self, file_path, grayscale=True, test_size=0.2, one_hot=True):
        """
        :param file_path: 文件路径, 我们这里应该为 trainImageList.txt 文件.
        :param grayscale: 布尔值, 默认为 True, 读取图片的格式, 为 True 时表示读取灰度图.
        :param test_size: 浮点数, 默认为 0.2, 将数据分割出训练集的比例.
        :param one_hot: 布尔值, 默认为 True, 独热编码.
        """
        self._file_path = file_path
        self._grayscale = grayscale
        self._test_size = test_size
        self._one_hot = one_hot
        self._labels = {'pos': np.array([1, 0, 0]),
                        'neg': np.array([0, 1, 0]),
                        'part': np.array([0, 0, 1])}

    def generate_12net_data(self, image, bounding_box):
        """
        给定一张图片, 生成 12*12 大小的 ROI 图片及其标签.
        实现思路:
        用任意大小的方形框从图像中切割出 ROI 子图, 计算该子图与 bounding_box 的重合度.
        由于一张图片中可能不止一个人脸的 bounding box, 所以在计算重合度时, 我们分别计算 ROI 与所有 box 的重合度.
        取重合度值最大的一个作分类的依据.
        :param test_size: 浮点数, 默认为 0.2, 将数据分割出训练集的比例.
        :return: [(image, label)]
        """
        m, n = image.shape[:2]
        roi_min_size = 12
        roi_max_size = np.max(m, n)
        roi_size = np.random.randint(roi_min_size, roi_max_size)

        return





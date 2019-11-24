"""
目标:
加载数据和生成数据.
我们的数据集中包含的是图片和其中人脸位置的 bounding box 标注. PNet 要求输入数据是 (None, 12, 12, 3).
也就是需要将原图像切割成 12*12 的大小 ROI 子图, 根据 bounding box 标注, 计算这个子图中是否包含人脸.
做一个分类训练, 分类结果为: pos, neg, part. 三种, 即: 全部为人脸部分, 没有人脸部分, 部分为人脸部分.

实现细节:
因为数据集给出的是对应图像中人脸的总数, 以及各个人脸的 bounding box.
而我们的目标是要生成 12*12 的 roi 子图, 并标注 pos, neg, part 三种标签.
所以选择, 神经网络训练的批次来自于一张图像, 我们从这张图像中截取 roi 子图生成这一批次的数据.
这样我们每一批次的图像数量是不一样的. 这应该没什么问题.
同样, 我们也可以规定在每一张图片中随机截取指定个数的 roi 子图. (我的选择).

参考链接:
https://github.com/LeslieZhoa/tensorflow-MTCNN/tree/master/preprocess

数据集介绍:
wider_face_split.zip
WIDER_test.zip
WIDER_train.zip
WIDER_val.zip
下载地址为: http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/
WIDER_test.zip
WIDER_train.zip
WIDER_val.zip
三个压缩包中都存放着图片.
wider_face_split.zip

以下是下载链接面页中的介绍.
WIDER FACE 数据集是一个面部检测基准数据集, 其图像是从可公开获得的WIDER数据集中选择的.
我们选择了 32,203 张图像, 并标记了 393,703 张在比例，姿势和遮挡方面具有高度可变性的面部. 如示例图像中所示.
WIDER FACE数据集基于 61 个事件类别进行组织. 对于每个事件类别, 我们随机选择 40％/ 10％/ 50％ 数据作为训练, 验证和测试集.
我们采用与 PASCAL VOC 数据集相同的评估指标. 与MALF和Caltech数据集相似, 我们不发布测试图像的边界框地面真相.
用户需要提交最终的预测文件, 我们将对其进行评估.
"""
import os

import cv2 as cv
import numpy as np

from project_example.mtcnn_face_detect_pnet.util import load_data_helper as ldh


class DataLoader(object):
    def __init__(self, file_path, gray_scale=True, test_size=0.2, one_hot=True):
        """
        :param file_path: 文件路径, 我们这里应该为 wider_face_train_bbx_gt.txt 文件.
        :param gray_scale: 布尔值, 默认为 True, 读取图片的格式, 为 True 时表示读取灰度图.
        :param test_size: 浮点数, 默认为 0.2, 将数据分割出训练集的比例.
        :param one_hot: 布尔值, 默认为 True, 现在只支持独热编码.
        """
        self._file_path = file_path
        self._gray_scale = gray_scale
        self._test_size = test_size
        self._one_hot = one_hot
        self._labels = {'pos': np.array([1, 0, 0]),
                        'neg': np.array([0, 1, 0]),
                        'part': np.array([0, 0, 1])}

        # 得到一个迭代器, 通过 next 方法来遍历图片, 及其标注信息.
        self._image_data = ldh.load_image_data(self._file_path)

    def generate_12net_data(self, image_data, quantity=100):
        """
        给定一张图片, 生成 12*12 大小的 ROI 图片及其标签.
        实现思路:
        用任意大小的方形框从图像中切割出 ROI 子图, 计算该子图与 bounding_box 的重合度.
        由于一张图片中可能不止一个人脸的 bounding box, 所以在计算重合度时, 我们分别计算 ROI 与所有 box 的重合度.
        取重合度值最大的一个作分类的依据.

        要促使生成平衡的数据, 我在标记的 bounding box 为中心的附近进行截取 ROI.
        给定一个标注的 bounding box, 取其长边设置 sigma, 按正态分布生成 box.

        图片路径的前半段: "dataset/WIDER_train"
        :param image_data: 输入图片相对路径及标注信息, 如:
        ['9--Press_Conference/9_Press_Conference_Press_Conference_9_591.jpg\n',
         2,
         [(473, 131, 32, 50), (759, 267, 10, 14)]]
        :param quantity: 指定需要从一张图片中随机抽取的 ROI 的数量.
        :return: [(roi, label)]

        demo:
        查看生成的样本中各类样本的数量.
        由于创建的迭代器是固定的, 所以可以执行一些 next(a) 之再查看后面的.
        在不同的图片中样本不平衡的比例不一样, 不过只要不是极其不平衡, 应该可以用吧.
        file_path = "dataset/wider_face_split/wider_face_train_bbx_gt.txt"
        data_loader = DataLoader(file_path=file_path)
        a = data_loader.batch_load_data()
        for i in range(10):
            next(a)
        data, target = next(a)
        print(np.sum(target, axis=0))
        """
        image_relative_path, face_count, bounding_box = image_data
        image_path = os.path.join("dataset/WIDER_train", image_relative_path)
        image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        m, n = image.shape
        roi_min_size = 12
        roi_max_size = np.min((m, n))
        roi_images = list()
        roi_labels = list()
        for i in range(quantity):
            index = np.random.randint(bounding_box.shape[0])
            x_, y_, w_, h_ = bounding_box[index]
            roi_size_ = np.max([w_, h_])
            # 三个方差, 分别用于调节三种标签的比例, 需要一个比较大的标签以在离人脸较远的地方截取 roi
            sigma = np.random.choice([roi_size_ / 20, roi_size_ / 5, roi_size_ * 2])
            x0, y0, s0 = np.random.normal(loc=0, scale=sigma, size=3)

            roi_size = np.clip(roi_size_ + int(s0), roi_min_size, roi_max_size)
            x = np.clip(x_ + int(x0), 0, n - roi_size)
            y = np.clip(y_ + int(y0), 0, m - roi_size)

            roi = np.resize(image[y:y+roi_size, x:x+roi_size], new_shape=(12, 12))

            selected_bounding_box = np.array([x, y, roi_size, roi_size])
            # 根据 bounding box IOU 重合程度, 计算标签.
            iou = ldh.bounding_box_iou(box=selected_bounding_box, boxes=bounding_box)
            if iou < 0.3:
                label = 'neg'
            elif iou < 0.7:
                label = 'part'
            else:
                label = 'pos'
            if self._one_hot:
                label = self._labels[label]
            roi_images.append(roi)
            roi_labels.append(label)
        data = np.stack(roi_images, axis=0)
        target = np.stack(roi_labels, axis=0)
        return data, target

    def batch_load_data(self, batch_size=100):
        """
        迭代器形式, 使用 for 循环来遍历数据, 当遇到 StopIteration 异常时, 会自动跳出循环.
        :param batch_size: 指定每一批次的数量. 在一个批次中, 我们只使用一个样本图片来生成这批次的数据.
        :return:

        demo:
        file_path = "dataset/wider_face_split/wider_face_train_bbx_gt.txt"
        data_loader = DataLoader(file_path=file_path)
        a = data_loader.batch_load_data()
        print(next(a))
        """
        while True:
            image_data = next(self._image_data)
            data, target = self.generate_12net_data(image_data, quantity=batch_size)
            yield data, target


if __name__ == '__main__':
    file_path = "dataset/wider_face_split/wider_face_train_bbx_gt.txt"
    data_loader = DataLoader(file_path=file_path)
    a = data_loader.batch_load_data()
    for i in range(12):
        next(a)
    data, target = next(a)
    print(np.sum(target, axis=0))

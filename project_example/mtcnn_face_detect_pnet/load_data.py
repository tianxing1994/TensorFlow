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



image_file_path = ''




"""
相关函数:
tf.io.TFRecordWriter
tf.train.Example
tf.train.Features
tf.train.Feature

example.SerializeToString()

tf.TFRecordReader
tf.train.string_input_producer
tf.parse_single_example
tf.FixedLenFeature
tf.decode_raw
tf.cast

tf.shuffle_batch
"""
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def demo1():
    path = 'jpg'
    filenames = os.listdir(path)
    writer = tf.io.TFRecordWriter("train.tfrecords")

    for name in os.listdir(path):
        class_path = path + os.sep + name
        for img_name in os.listdir(class_path):
            img_path = path + os.sep + name
            img = Image.open(img_path)
            img = img.resize((500, 500))
            img_raw = img.tobytes()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[name])),
                             "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}
                )
            )
            writer.write(example.SerializeToString())

    return


def demo2():
    """将图片文件转为二进制, 再存入 TFRecord 文件中"""
    image_path = r'C:\Users\Administrator\PycharmProjects\TensorFlow\datasets\cat.jpg'
    image = cv2.imread(image_path)

    image_raw = image.tobytes()

    # 不知道用什么方法, 取出 shape 中的多个值. 只能一个维度占一个位置了.
    shape0, shape1, shape2 = image.shape

    example = tf.train.Example(
        features=tf.train.Features(
            feature={"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[1])),
                     "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                     "shape0": tf.train.Feature(int64_list=tf.train.Int64List(value=[shape0])),
                     "shape1": tf.train.Feature(int64_list=tf.train.Int64List(value=[shape1])),
                     "shape2": tf.train.Feature(int64_list=tf.train.Int64List(value=[shape2])),
            }
        )
    )
    # print(example)
    # 将文件保存到当前路径下的 train.tfrecords 文件.
    writer = tf.io.TFRecordWriter("train.tfrecords")
    writer.write(example.SerializeToString())
    return


def demo3():
    """读取并解析 tfrecords 文件."""
    filename = "train.tfrecords"
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, example_serialized = reader.read(filename_queue)     # 返回文件名和文件

    features = tf.parse_single_example(example_serialized, features={
        "label": tf.FixedLenFeature([], tf.int64),
        "image": tf.FixedLenFeature([], tf.string),
        # 这个地方, 找不到一个办法取出 shape 中的多个值. 只能一个值存一个 key.
        "shape0": tf.FixedLenFeature([], tf.int64),
        "shape1": tf.FixedLenFeature([], tf.int64),
        "shape2": tf.FixedLenFeature([], tf.int64),
    })

    image = tf.decode_raw(features["image"], tf.uint8)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    threads = tf.train.start_queue_runners(sess=sess)

    image = tf.cast(image, tf.uint8)

    label = tf.cast(features['label'], tf.int32)
    shape0 = tf.cast(features['shape0'], tf.int32)
    shape1 = tf.cast(features['shape1'], tf.int32)
    shape2 = tf.cast(features['shape2'], tf.int32)

    imagecv2 = sess.run(image)
    shape = sess.run([shape0, shape1, shape2])

    imagecv2 = np.reshape(imagecv2, shape)

    cv2.imshow("window_name", imagecv2)
    cv2.waitKey(0)
    return


if __name__ == '__main__':
    demo2()
    demo3()


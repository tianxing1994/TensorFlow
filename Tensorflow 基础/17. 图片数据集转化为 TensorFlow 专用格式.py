"""
Dogs vs. Cats 数据集下载地址:
https://www.kaggle.com/c/dogs-vs-cats
"""
import re
import os
import cv2 as cv
import numpy as np
import tensorflow as tf


def show_image(image):
    cv.namedWindow('input image', cv.WINDOW_NORMAL)
    cv.imshow('input image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    """将图片数据写入 .tfrecords 文件"""
    train_data_path = 'E:/Python/dataset/dogs-vs-cats/train'
    test_data_path = 'E:/Python/dataset/dogs-vs-cats/test'
    train_tfrecords_path = 'E:/Python/dataset/dogs-vs-cats/train.tfrecords'
    test_tfrecords_path = 'E:/Python/dataset/dogs-vs-cats/test.tfrecords'

    writer = tf.io.TFRecordWriter(train_tfrecords_path)

    for root, sub_folder, files in os.walk(train_data_path):

        for file in files:
            filename = os.path.join(root, file)
            image = cv.imread(filename)
            shape = image.shape
            image_raw = image.tostring()
            pattern = re.compile('([a-z]+)\.')
            match = re.match(pattern, file)
            label = match.group(1)
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode('utf-8')])),
                    'shape0': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[0]])),
                    'shape1': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[1]])),
                    'shape2': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[2]])),
                }
            ))

            writer.write(example.SerializeToString())
            print('data write done: %s' % filename)
    writer.close()
    return


def demo2():
    """因为这里, 我们不能提前知道图片的形状, 所以在还原图片之前, 必须要先得到形状的值, 才能解析图片."""
    train_tfrecords_path = 'E:/Python/dataset/dogs-vs-cats/train.tfrecords'
    test_tfrecords_path = 'E:/Python/dataset/dogs-vs-cats/test.tfrecords'

    filename_queue = tf.train.string_input_producer([train_tfrecords_path])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'shape0': tf.FixedLenFeature([], tf.int64),
            'shape1': tf.FixedLenFeature([], tf.int64),
            'shape2': tf.FixedLenFeature([], tf.int64),
        }
    )

    image_raw = image_features['image_raw']
    label = image_features['label']
    shape0 = image_features['shape0']
    shape1 = image_features['shape1']
    shape2 = image_features['shape2']

    image_raw_batch, label_batch, shape0_batch, shape1_batch, shape2_batch = \
        tf.train.shuffle_batch([image_raw, label, shape0, shape1, shape2],
                               batch_size=10, min_after_dequeue=100, num_threads=64, capacity=200)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)

    image_raw_batch, label_batch, shape0_batch, shape1_batch, shape2_batch = \
        sess.run([image_raw_batch, label_batch, shape0_batch, shape1_batch, shape2_batch])
    shape_array = np.stack(arrays=(shape0_batch, shape1_batch, shape2_batch)).T

    for i in range(len(image_raw_batch)):
        image = tf.decode_raw(image_raw_batch[i], tf.uint8)
        image = sess.run(image)
        image = np.reshape(image, newshape=shape_array[i])
        print(label_batch[i].decode('utf-8'))
        show_image(image)
    return


def demo3():
    """该方法执行, 每次都返回的是第一张图片, 这是因为 TFRecordReader 在每次读取时,
    总是仅仅通过 Iterator 的方式读取当前队列的第一个元素, 其他元素在队列中进行等待.
    demo4 演示了怎样迭代地显示每一张图片.
    """
    train_tfrecords_path = 'E:/Python/dataset/dogs-vs-cats/train.tfrecords'
    test_tfrecords_path = 'E:/Python/dataset/dogs-vs-cats/test.tfrecords'

    filename_queue = tf.train.string_input_producer([train_tfrecords_path])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
            'shape0': tf.FixedLenFeature([], tf.int64),
            'shape1': tf.FixedLenFeature([], tf.int64),
            'shape2': tf.FixedLenFeature([], tf.int64),
        }
    )

    image_data = tf.decode_raw(image_features['image_raw'], tf.uint8)
    label = image_features['label']
    shape0 = image_features['shape0']
    shape1 = image_features['shape1']
    shape2 = image_features['shape2']

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)

    image_flatten, label, shape0, shape1, shape2 = sess.run([image_data, label, shape0, shape1, shape2])
    image = np.reshape(image_flatten, newshape=(shape0, shape1, shape2))
    print(label.decode('utf-8'))
    show_image(image)
    return


def demo4():
    """迭代地显示 tfrecords 文件中的图片."""
    train_tfrecords_path = 'E:/Python/dataset/dogs-vs-cats/train.tfrecords'
    test_tfrecords_path = 'E:/Python/dataset/dogs-vs-cats/test.tfrecords'

    for serialized_example in tf.io.tf_record_iterator(train_tfrecords_path):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        image_raw = example.features.feature['image_raw'].bytes_list.value
        label = example.features.feature['label'].bytes_list.value
        shape0 = example.features.feature['shape0'].int64_list.value
        shape1 = example.features.feature['shape1'].int64_list.value
        shape2 = example.features.feature['shape2'].int64_list.value

        shape = (shape0[0], shape1[0], shape2[0])
        label = label[0].decode('utf-8')

        image = tf.decode_raw(image_raw[0], tf.uint8)
        image = tf.reshape(image, shape=shape)

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        image = sess.run(image)

        print(label)
        show_image(image)
    return


def demo5():
    """将图片转换为固定形状大小, 再写入 .tfrecords 文件"""
    train_data_path = 'E:/Python/dataset/dogs-vs-cats/train'
    test_data_path = 'E:/Python/dataset/dogs-vs-cats/test'
    train_tfrecords_path = 'E:/Python/dataset/dogs-vs-cats/train-227.tfrecords'
    test_tfrecords_path = 'E:/Python/dataset/dogs-vs-cats/test-227.tfrecords'

    writer = tf.io.TFRecordWriter(train_tfrecords_path)

    for root, sub_folder, files in os.walk(train_data_path):

        for file in files:

            filename = os.path.join(root, file)
            image = cv.imread(filename)
            image = cv.resize(image, dsize=(227, 227))
            image_raw = image.tostring()
            pattern = re.compile('([a-z]+)\.')
            match = re.match(pattern, file)
            label = match.group(1)
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode('utf-8')])),
                }
            ))

            writer.write(example.SerializeToString())
            print('data write done: %s' % filename)
    writer.close()
    return


def demo6():
    """tf.train.shuffle_batch 读取数据, 这也就要求我们的数据集在读出之前必须为已知的相同大小. 暂时不知道这个应该怎么解决."""
    train_tfrecords_path = 'E:/Python/dataset/dogs-vs-cats/train-227.tfrecords'

    filename_queue = tf.train.string_input_producer([train_tfrecords_path])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string),
        }
    )

    image = tf.reshape(tf.decode_raw(image_features['image_raw'], tf.uint8), shape=(227, 227, 3))
    label = image_features['label']

    image_batch, label_batch = \
        tf.train.shuffle_batch([image, label], batch_size=10000, min_after_dequeue=100, num_threads=64, capacity=200)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)

    image_batch, label_batch = sess.run([image_batch, label_batch])
    print(type(image_batch))
    print(image_batch.shape)
    print(image_batch)

    print(type(label_batch))
    print(label_batch.shape)
    print(label_batch)

    show_image(image_batch[0])

    return


def demo7():
    """(我不知道它读出来的图片为什么是浮点数.), 直接将图片文件夹中的图片转化为 TensorFlow 格式以用于训练."""
    image_list = list()
    label_list = list()

    train_data_path = 'E:/Python/dataset/dogs-vs-cats/train'
    for root, sub_folder, files in os.walk(train_data_path):
        for file in files:
            filename = os.path.join(root, file)
            pattern = re.compile('([a-z]+)\.')
            match = re.match(pattern, file)
            label = match.group(1)
            image_list.append(filename)
            label_list.append(label)

    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.string)

    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, 227, 227)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=10, num_threads=64, capacity=200)
    label_batch = tf.reshape(label_batch, shape=(10,))

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)

    image_batch, label_batch = sess.run([image_batch, label_batch])

    print(type(image_batch))
    print(image_batch.shape)
    print(image_batch)

    print(type(label_batch))
    print(label_batch.shape)
    print(label_batch)

    show_image(image_batch[0])

    return


if __name__ == '__main__':
    demo6()


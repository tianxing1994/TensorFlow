"""
相关函数:
tf.train.Example
tf.train.Features
tf.train.Feature

tf.train.FloatList
tf.train.Int64List
tf.train.BytesList

tf.python_io.TFRecordWriter
tf.train.string_input_producer
tf.TFRecordReader
tf.parse_single_example
tf.decode_raw
tf.train.shuffle_batch
tf.train.start_queue_runners
"""
import tensorflow as tf
import numpy as np

def demo1():
    """创建 Example 对象"""
    a_data = 0.834
    b_data = [17]
    c_data = np.array([[0, 1, 2], [3, 4, 5]])
    c = c_data.astype(np.uint8)
    c_raw = c.tostring()

    example = tf.train.Example(
        features=tf.train.Features(
            feature={'a': tf.train.Feature(float_list=tf.train.FloatList(value=[a_data])),
                     'b': tf.train.Feature(int64_list=tf.train.Int64List(value=b_data)),
                     'c': tf.train.Feature(bytes_list=tf.train.BytesList(value=[c_raw]))}
        )
    )
    print(example)
    return


def demo2():
    """创建 tfrecords 文件, 并向其中写入内容"""
    writer = tf.python_io.TFRecordWriter("trainArray.tfrecords")
    for _ in range(100):
        randomArray = np.random.random((1, 3))
        # TFRecords 只能保存二进制数据.
        array_raw = randomArray.tobytes()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
                         "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[array_raw]))}
            )
        )
        writer.write(example.SerializeToString())
    writer.close()
    return


def demo3():
    """TFRecords 文件的读取"""
    filename_queue = tf.train.string_input_producer(["dataTest.tfrecords"], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'a': tf.FixedLenFeature([], tf.float32),
            'b': tf.FixedLenFeature([], tf.int64),
            'c': tf.FixedLenFeature([], tf.string)
        })

    a = features['a']
    b = features['b']
    c_raw = features['c']
    c = tf.decode_raw(c_raw, tf.uint8)
    c = tf.reshape(c, [2, 3])

    a_batch, b_batch, c_batch = tf.train.shuffle_batch([a, b, c], batch_size=1, capacity=200, min_after_dequeue=100, num_threads=2)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)
    a_val, b_val, c_val = sess.run([a_batch, b_batch, c_batch])
    print(a_val)
    print(b_val)
    print(c_val)

    return


def test():
    filename_queue = tf.train.string_input_producer([r"C:\Users\Administrator\PycharmProjects\TensorFlow\Tensorflow 基础\trainArray.tfrecords"], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    with tf.compat.v1.Session() as sess:
        result = sess.run(serialized_example)
        print(result)
    return


if __name__ == '__main__':
    test()

"""
相关函数:
tf.io.read_file
tf.io.decode_jpeg
tf.image.convert_image_dtype
"""
import tensorflow as tf


def demo1():
    file = tf.io.read_file('../datasets/cat.jpg')

    with tf.Session() as sess:
        result = sess.run(file)
        print(result)
    return


def demo2():
    encoded_image = tf.io.read_file('../datasets/cat.jpg')
    decoded_image = tf.io.decode_jpeg(encoded_image, channels=1)

    with tf.Session() as sess:
        decoded_image = sess.run(decoded_image)
        print(decoded_image)
        print(decoded_image.dtype)
    return


def demo3():
    encoded_image = tf.io.read_file('../datasets/cat.jpg')
    decoded_image = tf.io.decode_jpeg(encoded_image)
    converted_image = tf.image.convert_image_dtype(decoded_image, dtype=tf.float32)

    with tf.Session() as sess:
        converted_image = sess.run(converted_image)

        print(converted_image)
        print(converted_image.dtype)
    return


if __name__ == '__main__':
    demo2()

# 添加环境变量.
import os
import sys
p = os.getcwd()
sys.path.append(p)

import cv2 as cv
import numpy as np
import tensorflow as tf

from load_data import generate_training_set
from modeling import get_model


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    x_train, y_train = generate_training_set(n_images=1,
                                             image_size=32,
                                             min_object_size=2,
                                             max_object_size=16)
    # 训练数据转换成 BGR 图片
    image = np.array(np.squeeze(x_train) * 255, dtype=np.uint8)
    image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    show_image(image)

    model = get_model()
    ckpt_path = "model/box_regression.ckpt-1000"
    x = model.x
    y_pred = model.neural_networks(x=x)
    saver = tf.train.Saver(max_to_keep=3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_path)
        print("Model restored from file: %s" % ckpt_path)
        result = sess.run([y_pred], feed_dict={x: x_train})

    bounding_box = np.squeeze(result)
    x, y, w, h = np.array(np.round(bounding_box, decimals=0), dtype=np.uint8)
    print("predict bounding box: ", bounding_box)
    print("draw bounding box: ", x, y, w, h)
    print("real bounding box: ", y_train)
    cv.rectangle(image, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=1)
    show_image(image)
    return


if __name__ == '__main__':
    demo1()

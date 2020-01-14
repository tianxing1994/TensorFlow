# 添加环境变量.
import os
import sys
p = os.getcwd()
sys.path.append(p)

import cv2 as cv
import numpy as np
import tensorflow as tf

from load_data import batch_load_data
from modeling import get_model


def show_image(image, win_name='input image'):
    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return


def demo1():
    model = get_model()
    ckpt_path = "model/box_regression.ckpt-76"
    x = model.x
    y_pred = model.neural_networks(x=x)

    sess= tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=3)
    saver.restore(sess, ckpt_path)
    print("Model restored from file: %s" % ckpt_path)

    for x_train, y_train in batch_load_data(batch_size=1, flag="train"):
        image = np.array(np.squeeze(x_train), dtype=np.uint8)
        image_h, image_w, _ = image.shape
        show_image(image)

        result = sess.run([y_pred], feed_dict={x: x_train})

        bounding_box = np.squeeze(result)
        x_, y_, w_, h_ = bounding_box
        box_x = int(x_ * image_w)
        box_y = int(y_ * image_h)
        box_w = int(w_ * image_w)
        box_h = int(h_ * image_h)

        print("predict bounding box: ", x_, y_, w_, h_)
        print("draw bounding box: ", box_x, box_y, box_w, box_h)
        print("real bounding box: ", y_train)
        cv.rectangle(image, pt1=(box_x, box_y), pt2=(box_x+box_w, box_y+box_h), color=(0, 0, 255), thickness=1)
        show_image(image)
        # break
    return


if __name__ == '__main__':
    demo1()

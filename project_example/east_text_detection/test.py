# 添加环境变量.
import os
import sys
p = os.getcwd()
sys.path.append(p)


import time
import datetime
import collections
import logging

import cv2
import numpy as np
import tensorflow as tf
import model
# from icdar import restore_rectangle
# import lanms
from eval import resize_image, sort_poly, detect


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def draw_illu(illu, rst):
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu


# ckpt 文件的路径.
# checkpoint_path = './model'
checkpoint_path = './local_model'

input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

f_score, f_geometry = model.model(input_images, is_training=False)

variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
saver = tf.train.Saver(variable_averages.variables_to_restore())

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
logger.info('Restore from {}'.format(model_path))
saver.restore(sess, model_path)


def predictor(img):
    """
    :return: {
        'text_lines': [
            {
                'score': ,
                'x0': ,
                'y0': ,
                'x1': ,
                ...
                'y3': ,
            }
        ],
        'rtparams': {  # runtime parameters
            'image_size': ,
            'working_size': ,
        },
        'timing': {
            'net': ,
            'restore': ,
            'nms': ,
            'cpuinfo': ,
            'meminfo': ,
            'uptime': ,
        }
    }
    """
    start_time = time.time()
    rtparams = collections.OrderedDict()
    rtparams['start_time'] = datetime.datetime.now().isoformat()
    rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
    timer = collections.OrderedDict([
        ('net', 0),
        ('restore', 0),
        ('nms', 0)
    ])

    im_resized, (ratio_h, ratio_w) = resize_image(img)
    rtparams['working_size'] = '{}x{}'.format(
        im_resized.shape[1], im_resized.shape[0])
    start = time.time()
    score, geometry = sess.run(
        [f_score, f_geometry],
        feed_dict={input_images: [im_resized[:, :, ::-1]]})
    timer['net'] = time.time() - start

    boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
    logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
        timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

    if boxes is not None:
        scores = boxes[:, 8].reshape(-1)
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

    duration = time.time() - start_time
    timer['overall'] = duration
    logger.info('[timing] {}'.format(duration))

    text_lines = []
    if boxes is not None:
        text_lines = []
        for box, score in zip(boxes, scores):
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            tl = collections.OrderedDict(zip(
                ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                map(float, box.flatten())))
            tl['score'] = float(score)
            text_lines.append(tl)
    ret = {
        'text_lines': text_lines,
        'rtparams': rtparams,
        'timing': timer,
    }
    # ret.update(get_host_info())
    return ret


if __name__ == '__main__':
    image_path = 'snapshot_1572330104.jpg'
    image = cv2.imread(image_path)
    # image = image[:, :, ::-1]
    rst = predictor(image)
    draw_illu(image, rst)
    cv2.namedWindow(winname='input image', flags=cv2.WINDOW_NORMAL)
    cv2.imshow(winname='input image', mat=image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


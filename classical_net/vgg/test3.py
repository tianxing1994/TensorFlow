import cv2 as cv
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.models.research.slim.nets import vgg

ckpt_path = '../../datasets/classical_model/vgg_16.ckpt'
img_path = '../../datasets/image/cat.jpg'

vgg16 = vgg.vgg_d
inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name='inputs')
with slim.arg_scope(vgg.vgg_arg_scope()):
    net, _ = vgg16(inputs)
    softmax = tf.nn.softmax(net)
    predictions = tf.argmax(softmax, 1)

saver = tf.train.Saver(max_to_keep=3)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, ckpt_path)

    image = cv.imread(img_path)
    image = cv.resize(image, (224, 224))
    images = np.expand_dims(image, 0)
    ret = sess.run(predictions, feed_dict={inputs: images})
    print(ret)


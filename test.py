import tensorflow as tf


x = tf.constant([[1., 1.],
                 [2., 2.]])
m = tf.reduce_mean(x)

with tf.Session() as sess:
    result = sess.run(m)

print(type(result))



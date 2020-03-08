import tensorflow as tf


c = tf.constant([[1, 2, 3], [3, 2, 1]])
shape = tf.shape(c)[0]

sess = tf.Session()

ret = sess.run(shape)

print(ret)

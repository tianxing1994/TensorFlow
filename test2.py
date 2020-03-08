image = cv.imread(r'C:\Users\Administrator\PycharmProjects\TensorFlow\datasets\cat.jpg')
image = np.array(image, dtype=np.float32)
shape = image.shape
image = np.expand_dims(image, axis=0)

filter = tf.Variable(tf.ones([11, 11, 3, 1]))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    res = tf.nn.conv2d(image, filter, strides=[1, 2, 2, 1], padding="SAME")
    res_image = sess.run(tf.reshape(res, (int(shape[0] / 2), int(shape[1] / 2))))
    res_image = res_image / res_image.max()

    show_image(res_image)
"""
导入 VGG16 预训练模型参数.
https://www.jianshu.com/p/39e39fab0065
"""
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

vgg16_ckpt_path = '../../datasets/classical_model/vgg_16.ckpt'


def load_model(input, model_path=vgg16_ckpt_path):
    net = {}

    reader = pywrap_tensorflow.NewCheckpointReader(model_path)
    print(reader)
    vgg_variable = reader.get_variable_to_shape_map()
    keys = sorted(vgg_variable)

    # Print tensor name and values
    # for key in keys:
    #     if key > 'vgg_16/conv6':
    #         print("tensor_name: ", key,reader.get_tensor(key).shape)

    # conv1_1
    net['conv1_1'] = tf.nn.conv2d(input, reader.get_tensor('vgg_16/conv1/conv1_1/weights'), [1, 1, 1, 1],
                                  padding='SAME')
    net['conv1_1'] = tf.nn.bias_add(net['conv1_1'], reader.get_tensor('vgg_16/conv1/conv1_1/biases'))
    net['conv1_1'] = tf.nn.relu(net['conv1_1'])

    # conv1_2
    net['conv1_2'] = tf.nn.conv2d(net['conv1_1'], reader.get_tensor('vgg_16/conv1/conv1_2/weights'), [1, 1, 1, 1],
                                  padding='SAME')
    net['conv1_2'] = tf.nn.bias_add(net['conv1_2'], reader.get_tensor('vgg_16/conv1/conv1_2/biases'))
    net['conv1_2'] = tf.nn.relu(net['conv1_2'])

    # pool1
    net['pool1'] = tf.nn.max_pool(net['conv1_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv2_1
    net['conv2_1'] = tf.nn.conv2d(net['pool1'], reader.get_tensor('vgg_16/conv2/conv2_1/weights'), [1, 1, 1, 1],
                                  padding='SAME')
    net['conv2_1'] = tf.nn.bias_add(net['conv2_1'], reader.get_tensor('vgg_16/conv2/conv2_1/biases'))
    net['conv2_1'] = tf.nn.relu(net['conv2_1'])

    # conv2_2
    net['conv2_2'] = tf.nn.conv2d(net['conv2_1'], reader.get_tensor('vgg_16/conv2/conv2_2/weights'), [1, 1, 1, 1],
                                  padding='SAME')
    net['conv2_2'] = tf.nn.bias_add(net['conv2_2'], reader.get_tensor('vgg_16/conv2/conv2_2/biases'))
    net['conv2_2'] = tf.nn.relu(net['conv2_2'])

    # pool2
    net['pool2'] = tf.nn.max_pool(net['conv2_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv3_1
    net['conv3_1'] = tf.nn.conv2d(net['pool2'], reader.get_tensor('vgg_16/conv3/conv3_1/weights'), [1, 1, 1, 1],
                                  padding='SAME')
    net['conv3_1'] = tf.nn.bias_add(net['conv3_1'], reader.get_tensor('vgg_16/conv3/conv3_1/biases'))
    net['conv3_1'] = tf.nn.relu(net['conv3_1'])

    # conv3_2
    net['conv3_2'] = tf.nn.conv2d(net['conv3_1'], reader.get_tensor('vgg_16/conv3/conv3_2/weights'), [1, 1, 1, 1],
                                  padding='SAME')
    net['conv3_2'] = tf.nn.bias_add(net['conv3_2'], reader.get_tensor('vgg_16/conv3/conv3_2/biases'))
    net['conv3_2'] = tf.nn.relu(net['conv3_2'])

    # conv3_3
    net['conv3_3'] = tf.nn.conv2d(net['conv3_2'], reader.get_tensor('vgg_16/conv3/conv3_3/weights'), [1, 1, 1, 1],
                                  padding='SAME')
    net['conv3_3'] = tf.nn.bias_add(net['conv3_3'], reader.get_tensor('vgg_16/conv3/conv3_3/biases'))
    net['conv3_3'] = tf.nn.relu(net['conv3_3'])

    # pool3
    net['pool3'] = tf.nn.max_pool(net['conv3_3'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv4_1
    net['conv4_1'] = tf.nn.conv2d(net['pool3'], reader.get_tensor('vgg_16/conv4/conv4_1/weights'), [1, 1, 1, 1],
                                  padding='SAME')
    net['conv4_1'] = tf.nn.bias_add(net['conv4_1'], reader.get_tensor('vgg_16/conv4/conv4_1/biases'))
    net['conv4_1'] = tf.nn.relu(net['conv4_1'])

    # conv4_2
    net['conv4_2'] = tf.nn.conv2d(net['conv4_1'], reader.get_tensor('vgg_16/conv4/conv4_2/weights'), [1, 1, 1, 1],
                                  padding='SAME')
    net['conv4_2'] = tf.nn.bias_add(net['conv4_2'], reader.get_tensor('vgg_16/conv4/conv4_2/biases'))
    net['conv4_2'] = tf.nn.relu(net['conv4_2'])

    # conv4_3
    net['conv4_3'] = tf.nn.conv2d(net['conv4_2'], reader.get_tensor('vgg_16/conv4/conv4_3/weights'), [1, 1, 1, 1],
                                  padding='SAME')
    net['conv4_3'] = tf.nn.bias_add(net['conv4_3'], reader.get_tensor('vgg_16/conv4/conv4_3/biases'))
    net['conv4_3'] = tf.nn.relu(net['conv4_3'])

    # pool4
    net['pool4'] = tf.nn.max_pool(net['conv4_3'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # conv5_1
    net['conv5_1'] = tf.nn.conv2d(net['pool4'], reader.get_tensor('vgg_16/conv5/conv5_1/weights'), [1, 1, 1, 1],
                                  padding='SAME')
    net['conv5_1'] = tf.nn.bias_add(net['conv5_1'], reader.get_tensor('vgg_16/conv5/conv5_1/biases'))
    net['conv5_1'] = tf.nn.relu(net['conv5_1'])

    # conv5_2
    net['conv5_2'] = tf.nn.conv2d(net['conv5_1'], reader.get_tensor('vgg_16/conv5/conv5_2/weights'), [1, 1, 1, 1],
                                  padding='SAME')
    net['conv5_2'] = tf.nn.bias_add(net['conv5_2'], reader.get_tensor('vgg_16/conv5/conv5_2/biases'))
    net['conv5_2'] = tf.nn.relu(net['conv5_2'])

    # conv5_3
    net['conv5_3'] = tf.nn.conv2d(net['conv5_2'], reader.get_tensor('vgg_16/conv5/conv5_3/weights'), [1, 1, 1, 1],
                                  padding='SAME')
    net['conv5_3'] = tf.nn.bias_add(net['conv5_3'], reader.get_tensor('vgg_16/conv5/conv5_3/biases'))
    net['conv5_3'] = tf.nn.relu(net['conv5_3'])

    # pool5
    net['pool5'] = tf.nn.max_pool(net['conv5_3'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    p5_shape = net['pool5'].shape
    # print(net['pool5'].shape)
    net['reshape'] = tf.reshape(net['pool5'], shape=[-1, p5_shape[1] * p5_shape[2] * p5_shape[3]])

    # fc6
    net['fc6'] = tf.matmul(net['reshape'], tf.reshape(reader.get_tensor('vgg_16/fc6/weights'),
                                                      shape=[p5_shape[1] * p5_shape[2] * p5_shape[3], 4096]))
    net['fc6'] = tf.add(net['fc6'], reader.get_tensor('vgg_16/fc6/biases'))
    net['fc6'] = tf.nn.relu(tf.nn.dropout(net['fc6'], keep_prob=0.5))

    # fc7
    net['fc7'] = tf.matmul(net['fc6'], tf.reshape(reader.get_tensor('vgg_16/fc7/weights'),
                                                  shape=[4096, 4096]))
    net['fc7'] = tf.add(net['fc7'], reader.get_tensor('vgg_16/fc7/biases'))
    net['fc7'] = tf.nn.relu(tf.nn.dropout(net['fc7'], keep_prob=0.5))

    # fc8
    net['fc8'] = tf.matmul(net['fc7'], tf.reshape(reader.get_tensor('vgg_16/fc8/weights'),
                                                  shape=[4096, 1000]))
    net['fc8'] = tf.add(net['fc8'], reader.get_tensor('vgg_16/fc8/biases'))
    net['fc8'] = tf.nn.relu(net['fc8'])

    softmax = tf.nn.softmax(net['fc8'])
    predictions = tf.argmax(softmax, 1)
    return net, predictions


def main():
    """
    ImageNet 1000 个类别的对应表.
    https://blog.csdn.net/weixin_41770169/article/details/80482942
    :return:
    """
    img_path = '../../datasets/image/panda.jpg'
    with tf.Graph().as_default():
        with tf.Session() as sess:
            img_bytes = tf.read_file(img_path)
            png = img_path.lower().endswith('png')
            image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
            image = tf.image.resize_images(image, [224, 224], method=0)
            image = tf.to_float(sess.run(image))
            image = tf.expand_dims(image, 0)
            print(image.shape)
            if image.shape != (1, 224, 224, 3):
                print("图像大小不符合224 x 224 x 3")
            net, predictions = load_model(image)
            sess.run(tf.global_variables_initializer())
            print(sess.run(predictions))


if __name__ == '__main__':
    main()

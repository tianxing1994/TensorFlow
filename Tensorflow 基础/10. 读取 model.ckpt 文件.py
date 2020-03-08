"""
从 model.ckpt 文件中读取出训练好的权重值 (即网络中 Variable 的值).
如果在训练时, 没有给 Variable 变量指定 name 参数. 则读取到的变量名是自动生成的, 即: Variable, Variable_1, Variable_2 ......
如果给 Variable 指定了 name 名称, 则每一个被存储的权重, 的名称都是被指定的名称.
"""
import os
import numpy as np

import tensorflow as tf


def demo1():
    """从 model.ckpt 文件中读取出训练好的权重值"""
    checkpoint_path = "C:/Users/Administrator/PycharmProjects/TensorFlow/datasets/models/LeNet_model/model.ckpt"
    reader = tf.train.NewCheckpointReader(checkpoint_path)

    # 得到所有的权重名称及其对应的形状. 如果在训练模型时为每个变量设置了 name 参数, 则此处的名称会是那些参数值.
    # 如:
    # {'b_fc1': [80], 'b_fc2': [10], 'w_fc1': [5880, 80], 'filter2': [5, 5, 6, 16],
    # 'bias1': [6], 'filter1': [5, 5, 1, 6], 'bias2': [16], 'bias3': [120],
    # 'filter3': [5, 5, 16, 120], 'w_fc2': [80, 10]}
    all_variables = reader.get_variable_to_shape_map()
    print(all_variables)

    # 通过张量名, 获取张量值.
    print(reader.get_tensor("filter1"))
    print(reader.get_tensor("bias1"))
    print(reader.get_tensor("filter2"))
    print(reader.get_tensor("bias2"))
    print(reader.get_tensor("filter3"))
    print(reader.get_tensor("bias3"))
    print(reader.get_tensor("w_fc1"))
    print(reader.get_tensor("b_fc1"))
    print(reader.get_tensor("w_fc2"))
    print(reader.get_tensor("b_fc2"))

    # b'b_fc1 (DT_FLOAT) [80]\n
    # b_fc2 (DT_FLOAT) [10]\n
    # bias1 (DT_FLOAT) [6]\n
    # bias2 (DT_FLOAT) [16]\n
    # bias3 (DT_FLOAT) [120]\n
    # filter1 (DT_FLOAT) [5,5,1,6]\n
    # filter2 (DT_FLOAT) [5,5,6,16]\n
    # filter3 (DT_FLOAT) [5,5,16,120]\n
    # w_fc1 (DT_FLOAT) [5880,80]\n
    # w_fc2 (DT_FLOAT) [80,10]\n'
    result = reader.debug_string()
    print(result)
    print(type(result))

    # 获取每一个权重对应的数据类型. 如:
    # {'b_fc1': tf.float32, 'b_fc2': tf.float32, 'w_fc1': tf.float32, 'filter2': tf.float32,
    # 'bias1': tf.float32, 'filter1': tf.float32, 'bias2': tf.float32, 'bias3': tf.float32,
    # 'filter3': tf.float32, 'w_fc2': tf.float32}
    result = reader.get_variable_to_dtype_map()
    print(result)

    # 检测某个张量是否存在.
    result = reader.has_tensor("filter3")
    print(result)

    return


def demo2():
    """还有关于, 查看网络结构, 等, 还不太明白."""
    return


if __name__ == '__main__':
    demo1()


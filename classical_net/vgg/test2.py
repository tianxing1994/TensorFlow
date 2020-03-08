import tensorflow as tf
import pprint   # 使用pprint 提高打印的可读性

vgg16_ckpt_path = '../../datasets/classical_model/vgg_16.ckpt'
NewCheck =tf.train.NewCheckpointReader(vgg16_ckpt_path)

# 输出 ckpt 中所有变量信息, 三个字段: 名字, 数据类型, shape
print("debug_string:\n")
pprint.pprint(NewCheck.debug_string().decode("utf-8"))

# 打印张量中的值.
print("get_tensor:\n")
pprint.pprint(NewCheck.get_tensor("vgg_16/conv2/conv2_1/biases"))

print("get_variable_to_dtype_map\n")
pprint.pprint(NewCheck.get_variable_to_dtype_map())
print("get_variable_to_shape_map\n")
pprint.pprint(NewCheck.get_variable_to_shape_map())


"""
参考链接:
https://blog.csdn.net/gzroy/article/details/82386540
https://www.jianshu.com/p/23c73b90657f
https://blog.csdn.net/u013841196/article/details/80713314

通过分析, 我总结出 resnet_18 的网络结构如下: (不一定正确).
<-------------------------------------------------------------------- resnet_v1_18: --------------------------------------------------------------------->
1. 每一个 conv* 层, 对应为 conv2d, batch_normalization, relu.
2. 每个 add*_* 层, 对应 add, relu.
3. shortcut_* 时, 如果之前的深度和之后的深度相同, 只是 hight, width, 不同, 则采用 max_pool2d, 否则采用 conv2d. (在参考链接中说这之后还要 batch_normalization, 但在 resnet_v1 源码中好像没有).
4. inputs 的输入大小应为: None×224×224×channels.
5. conv0 + convi_i(4*4) + fc_1 = 18
images -> [batch, height, width, channels]

<------------------------------------------------------------------------------ init: ------------------------------------------------------------------------------------------------------------>
[name]             [input]                            [output]                         [action]        [output_size]                       [conv kernel size]                     [stride]    [pad]
conv0,             input=images,                      output=conv0,                    convolution,    output_size=None×112×112×channels,  kernel_size=7×7, filters=64,           stride=2,   SAME,
pool0,             input=conv0,                       output=pool0,                    max_pool2d,     output_size=None×56×56×channels,    kernel_size=3×3, filters=64,           stride=2

<------------------------------------------------------------------------------ stage_1: --------------------------------------------------------------------------------------------------------->
[block_0]:
conv1_0,           input=pool0,                       output=conv1_0,                  convolution,    output_size=None×56×56×channels,    kernel_size=3×3, filters=64,           stride=1,   SAME,
conv1_1,           input=conv1_0,                     output=conv1_1,                  convolution,    output_size=None×56×56×channels,    kernel_size=3×3, filters=64,           stride=1,   SAME,
add1_0,            input=conv1_1+pool0,               output=add1_0,                   add,            output_size=None×56×56×channels,

[block_1]:
conv1_2,           input=add1_0,                      output=conv1_2,                  convolution,    output_size=None×56×56×channels,    kernel_size=3×3, filters=64,           stride=1,   SAME,
conv1_3,           input=conv1_2,                     output=conv1_3,                  convolution,    output_size=None×56×56×channels,    kernel_size=3×3, filters=64,           stride=1,   SAME,
add1_1,            input=conv1_3+add1_0,              output=add1_1,                   add,            output_size=None×56×56×channels,

<------------------------------------------------------------------------------ stage_2: --------------------------------------------------------------------------------------------------------->
[block_0]:
shortcut2,         input=add1_1,                      output=shortcut2,                convolution,    output_size=None×28×28×channels,    kernel_size=1×1, filters=128,          stride=2,   SAME,
conv2_0,           input=add1_1,                      output=conv2_0,                  convolution,    output_size=None×28×28×channels,    kernel_size=3×3, filters=128,          stride=2,   SAME,
conv2_1,           input=conv2_0,                     output=conv2_1,                  convolution,    output_size=None×28×28×channels,    kernel_size=3×3, filters=128,          stride=1,   SAME,
add2_0,            input=conv2_1+shortcut2,           output=add2_0,                   add,            output_size=None×28×28×channels,

[block_1]:
conv2_2,           input=add2_0,                      output=conv2_2,                  convolution,    output_size=None×28×28×channels,    kernel_size=3×3, filters=128,          stride=1,   SAME,
conv2_3,           input=conv2_2,                     output=conv2_3,                  convolution,    output_size=None×28×28×channels,    kernel_size=3×3, filters=128,          stride=1,   SAME,
add2_1,            input=conv2_3+add2_0,              output=add2_1,                   convolution,    output_size=None×28×28×channels,

<------------------------------------------------------------------------------ stage_3: --------------------------------------------------------------------------------------------------------->
[block_0]:
shortcut3,         input=add2_1,                      output=shortcut3,                convolution,    output_size=None×14×14×channels,    kernel_size=1×1, filters=256,          stride=2,   SAME,
conv3_0,           input=add2_1,                      output=conv3_0,                  convolution,    output_size=None×14×14×channels,    kernel_size=3×3, filters=256,          stride=2,   SAME,
conv3_1,           input=conv3_0,                     output=conv3_1,                  convolution,    output_size=None×14×14×channels,    kernel_size=3×3, filters=256,          stride=1,   SAME,
add3_0,            input=conv3_1+shortcut3,           output=add3_0,                   add,            output_size=None×14×14×channels,

[block_1]:
conv3_2,           input=add3_0,                      output=conv3_2,                  convolution,    output_size=None×14×14×channels,    kernel_size=3×3, filters=256,          stride=1,   SAME,
conv3_3,           input=conv3_2,                     output=conv3_3,                  convolution,    output_size=None×14×14×channels,    kernel_size=3×3, filters=256,          stride=1,   SAME,
add3_1,            input=conv3_3+add3_0,              output=add3_1,                   add,            output_size=None×14×14×channels,

<------------------------------------------------------------------------------ stage_4: --------------------------------------------------------------------------------------------------------->
[block_0]:
shortcut4,         input=add3_1,                      output=shortcut4,                convolution,    output_size=None×7×7×channels,      kernel_size=1×1, filters=512,          stride=2,   SAME,
conv4_0,           input=add3_1,                      output=conv4_0,                  convolution,    output_size=None×7×7×channels,      kernel_size=3×3, filters=512,          stride=2,   SAME,
conv4_1,           input=conv4_0,                     output=conv4_1,                  convolution,    output_size=None×7×7×channels,      kernel_size=3×3, filters=512,          stride=1,   SAME,
add4_0,            input=conv4_1+shortcut4,           output=add4_0,                   add,            output_size=None×7×7×channels,

[block_1]:
conv4_2,           input=add4_0,                      output=conv4_2,                  convolution,    output_size=None×7×7×channels,      kernel_size=3×3, filters=512,          stride=1,   SAME,
conv4_3,           input=conv4_2,                     output=conv4_3,                  convolution,    output_size=None×7×7×channels,      kernel_size=3×3, filters=512,          stride=1,   SAME,
add4_1,            input=conv4_3+add4_0,              output=add4_1,                   add,            output_size=None×7×7×channels,
<------------------------------------------------------------------------------ average_pool: ---------------------------------------------------------------------------------------------------->
pool5:             input=add4_1,                      output=pool5,                    reduce_mean,    output_size=None×1×1×channels,      "在 height, width 两个轴求平均值, 并 keep_dim=True."

<------------------------------------------------------------------------------ fc_1: ------------------------------------------------------------------------------------------------------------>
fc_0,              input=pool5,                       output=fc_1,                     convolution,    output_size=1×1×num_classes,        kernel_size=1×1, filters=num_classes,  stride=1,    SAME,

<------------------------------------------------------------------------------ softmax: --------------------------------------------------------------------------------------------------------->
...
<------------------------------------------------------------------------------ end: ------------------------------------------------------------------------------------------------------------->


此处练习, 我们需要对 cifar10 的数据进行分类, 但 cifar10 的图片大小不足 224×224, 所以对网络会稍有不同.

1 + 3 * 5 * 2 + 1 = 32 个卷积层.

input:         None × 32 × 32 ×   3
init:          None × 32 × 32 ×  16
stage1:       [None × 32 × 32 ×  16] × 5
stage2:       [None × 16 × 16 ×  32] × 5
stage3:       [None × 8  × 8  ×  64] × 5
reduce_mean:  None × 1  × 1  ×  64

fc:           None × 10

softmax:
"""
import tensorflow as tf


def _resnet_block_v1(inputs, filters, projection, n_stage, n_block, training):
    n_stage += 1
    # 一个残差网络的 block 基本单位.
    stride = 2 if projection else 1
    # 在 stage>0 且 block=0 时执行 projection=True, stride=2.
    if projection:
        shortcut = tf.layers.conv2d(inputs, filters, (1, 1), strides=(stride, stride),
                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                    reuse=tf.AUTO_REUSE, padding="SAME", data_format='channels_last',
                                    name=f'conv_shortcut_{n_stage}')
        print(f'conv_shortcut_{n_stage}: {shortcut.shape}')
        shortcut = tf.layers.batch_normalization(shortcut, axis=3, training=training, reuse=tf.AUTO_REUSE,
                                                 name=f'bn_shortcut_{n_stage}')
    else:
        shortcut = inputs
    outputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=(3, 3), strides=(stride, stride),
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                               reuse=tf.AUTO_REUSE, padding="SAME", data_format='channels_last',
                               name=f'conv{n_stage}_{n_block*2}')
    print(f'conv{n_stage}_{n_block*2}: {outputs.shape}')

    outputs = tf.layers.batch_normalization(outputs, axis=3, training=training, reuse=tf.AUTO_REUSE,
                                            name=f'bn{n_stage}_{n_block*2}')
    outputs = tf.nn.relu(outputs, name=f'relu{n_stage}_{n_block*2}')

    outputs = tf.layers.conv2d(outputs, filters, kernel_size=(3, 3), strides=(1, 1),
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                               reuse=tf.AUTO_REUSE, padding="SAME", data_format='channels_last',
                               name=f'conv{n_stage}_{n_block*2+1}')
    print(f'conv{n_stage}_{n_block*2+1}: {outputs.shape}')

    outputs = tf.layers.batch_normalization(outputs, axis=3, training=training, reuse=tf.AUTO_REUSE,
                                            name=f'bn{n_stage}_{n_block*2+1}')

    outputs = tf.add(shortcut, outputs, name=f'add{n_stage}_{n_block}')
    print(f'add{n_stage}_{n_block}: {outputs.shape}')
    outputs = tf.nn.relu(outputs, name=f'relu{n_stage}_{n_block*2+1}')
    return outputs


def resnet_v1_32(data, training):
    # 针对 cifar10 数据集, None*3*32*32 的数据, 输出 None*10, 10 个类别.
    filters = 16
    # reuse 参数的意思是, 比如: 如果此次 conv2d 操作的 name='conv1', 则它首先会查找是否有 name='conv1' 的操作已存在, 如果存在, 则 reuse 已有的参数.
    inputs = tf.layers.conv2d(data, filters=filters, kernel_size=(3, 3), strides=(1, 1),
                              reuse=tf.AUTO_REUSE, padding='SAME', data_format='channels_last', name='conv0')
    print(f'conv0: {inputs.shape}')
    inputs = tf.layers.batch_normalization(inputs, axis=3, training=training,
                                           reuse=tf.AUTO_REUSE, name='bn_conv0')
    inputs = tf.nn.relu(inputs, name='relu0')

    for n_stage in range(3):
        stage_filter = filters * (2 ** n_stage)
        for n_block in range(5):
            projection = False
            if n_block == 0 and n_stage > 0:
                projection = True
            inputs = _resnet_block_v1(inputs, stage_filter, projection, n_stage, n_block, training=training)

    # 平均值池化.
    inputs = tf.reduce_mean(inputs, [1, 2])
    inputs = tf.identity(inputs, name='pool5')  # 为上一步的 reduce_mean 操作添加一个 name.
    print(f'pool5: {inputs.shape}')
    inputs = tf.layers.dense(inputs=inputs, units=10, reuse=tf.AUTO_REUSE, name='fc0')
    print(f'fc0: {inputs.shape}')
    return inputs




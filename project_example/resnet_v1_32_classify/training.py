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
conv0,             input=images,                      output=conv1,                    convolution,    output_size=None×112×112×channels,  kernel_size=7×7, filters=64,           stride=2,   SAME,
pool0,             input=conv0,                       output=pool0,                    max_pool2d,     output_size=None×56×56×channels,    kernel_size=3×3, filters=64,           stride=2

<------------------------------------------------------------------------------ stage_1: --------------------------------------------------------------------------------------------------------->
[block_0]:
conv1_1,           input=pool0,                       output=conv1_1,                  convolution,    output_size=None×56×56×channels,    kernel_size=3×3, filters=64,           stride=1,   SAME,
conv1_2,           input=conv1_1,                     output=conv1_2,                  convolution,    output_size=None×56×56×channels,    kernel_size=3×3, filters=64,           stride=1,   SAME,
add1_1,            input=conv1_2+pool0,               output=add1_1,                   add,            output_size=None×56×56×channels,

[block_1]:
conv1_3,           input=add1_1,                      output=conv1_3,                  convolution,    output_size=None×56×56×channels,    kernel_size=3×3, filters=64,           stride=1,   SAME,
conv1_4,           input=conv1_3,                     output=conv1_4,                  convolution,    output_size=None×56×56×channels,    kernel_size=3×3, filters=64,           stride=1,   SAME,
add1_2,            input=conv1_4+add1_1,              output=add1_2,                   add,            output_size=None×56×56×channels,

<------------------------------------------------------------------------------ stage_2: --------------------------------------------------------------------------------------------------------->
[block_0]:
shortcut2,         input=add1_2,                      output=shortcut2,                convolution,    output_size=None×28×28×channels,    kernel_size=1×1, filters=128,          stride=2,   SAME,
conv2_1,           input=add1_2,                      output=conv2_1,                  convolution,    output_size=None×28×28×channels,    kernel_size=3×3, filters=128,          stride=2,   SAME,
conv2_2,           input=conv2_1,                     output=conv2_2,                  convolution,    output_size=None×28×28×channels,    kernel_size=3×3, filters=128,          stride=1,   SAME,
add2_1,            input=conv2_2+shortcut2,           output=add2_1,                   add,            output_size=None×28×28×channels,

[block_1]:
conv2_3,           input=add2_1,                      output=conv2_3,                  convolution,    output_size=None×28×28×channels,    kernel_size=3×3, filters=128,          stride=1,   SAME,
conv2_4,           input=conv2_3,                     output=conv2_4,                  convolution,    output_size=None×28×28×channels,    kernel_size=3×3, filters=128,          stride=1,   SAME,
add2_2,            input=conv2_4+add2_1,              output=add2_2,                   convolution,    output_size=None×28×28×channels,

<------------------------------------------------------------------------------ stage_3: --------------------------------------------------------------------------------------------------------->
[block_0]:
shortcut3,         input=add2_2,                      output=shortcut3,                convolution,    output_size=None×14×14×channels,    kernel_size=1×1, filters=256,          stride=2,   SAME,
conv3_1,           input=add2_2,                      output=conv3_1,                  convolution,    output_size=None×14×14×channels,    kernel_size=3×3, filters=256,          stride=2,   SAME,
conv3_2,           input=conv3_1,                     output=conv3_2,                  convolution,    output_size=None×14×14×channels,    kernel_size=3×3, filters=256,          stride=1,   SAME,
add3_1,            input=conv3_2+shortcut3,           output=add3_1,                   add,            output_size=None×14×14×channels,

[block_1]:
conv3_1,           input=add3_1,                      output=conv3_1,                  convolution,    output_size=None×14×14×channels,    kernel_size=3×3, filters=256,          stride=1,   SAME,
conv3_2,           input=conv3_1,                     output=conv3_2,                  convolution,    output_size=None×14×14×channels,    kernel_size=3×3, filters=256,          stride=1,   SAME,
add3_2,            input=conv3_2+add3_1,              output=add3_2,                   add,            output_size=None×14×14×channels,

<------------------------------------------------------------------------------ stage_4: --------------------------------------------------------------------------------------------------------->
[block_0]:
shortcut4,         input=add3_2,                      output=shortcut4,                convolution,    output_size=None×7×7×channels,      kernel_size=1×1, filters=512,          stride=2,   SAME,
conv4_1,           input=add3_2,                      output=conv4_1,                  convolution,    output_size=None×7×7×channels,      kernel_size=3×3, filters=512,          stride=2,   SAME,
conv4_2,           input=conv4_1,                     output=conv4_2,                  convolution,    output_size=None×7×7×channels,      kernel_size=3×3, filters=512,          stride=1,   SAME,
add4_1,            input=conv4_2+shortcut4,           output=add4_1,                   add,            output_size=None×7×7×channels,

[block_1]:
conv4_3,           input=add4_1,                      output=conv4_3,                  convolution,    output_size=None×7×7×channels,      kernel_size=3×3, filters=512,          stride=1,   SAME,
conv4_4,           input=conv4_3,                     output=conv4_4,                  convolution,    output_size=None×7×7×channels,      kernel_size=3×3, filters=512,          stride=1,   SAME,
add4_2,            input=conv4_4+add4_1,              output=add4_2,                   add,            output_size=None×7×7×channels,
<------------------------------------------------------------------------------ average_pool: ---------------------------------------------------------------------------------------------------->
pool5:             input=add4_2,                      output=pool5,                    reduce_mean,    output_size=None×1×1×channels,      "在 height, width 两个轴求平均值, 并 keep_dim=True."

<------------------------------------------------------------------------------ fc_1: ------------------------------------------------------------------------------------------------------------>
fc_1,              input=pool5,                       output=fc_1,                     convolution,    output_size=1×1×num_classes,        kernel_size=1×1, filters=num_classes,  stride=1,    SAME,

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

from resnet_v1_32 import resnet_v1_32
from cifar10 import batch_load_cifar10_train


def sparse_accuarcy(labels, logits):
    labels = tf.cast(labels, dtype=tf.int64)
    logits = tf.math.argmax(input=logits, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, logits), tf.float64))
    return accuracy


x = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3))
y = tf.placeholder(dtype=tf.int32, shape=(None,))
logits = resnet_v1_32(data=x, training=True)

cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits,
                                                       reduction=tf.losses.Reduction.MEAN)

l2_loss = 2e-4 * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
tf.summary.scalar('l2_loss', l2_loss)
loss = cross_entropy + l2_loss
loss = tf.identity(loss, name='cross_entropy')

accuarcy = sparse_accuarcy(labels=y, logits=logits)

learning_rate = 1e-3
# AdamOptimizer 应该在 sess.run(tf.global_variables_initializer()) 前面, 否则报错.
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=3)

batch_size = 1000
epoch = 100
step = 0

for data, labels in batch_load_cifar10_train(batch_size=batch_size, epoch=epoch):
    lossValue, theAccuarcy, _ = sess.run([loss, accuarcy, train_op], feed_dict={x: data, y: labels})
    if step % 100 == 0:
        print("step %i, theAccuarcy %f, Loss: %f" %(step, theAccuarcy, lossValue))
        saver.save(sess, "model/resnet_v1_32/resnet_v1_32.ckpt", global_step=step)
    step += 1




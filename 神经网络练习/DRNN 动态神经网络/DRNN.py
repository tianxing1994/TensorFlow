import random

# ====================
#  TOY DATA GENERATOR
# ====================
class ToySequenceData(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])

    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """

    # 产生的数据类型为包含浮点数的二层列表.
    # data: 大致长度为 max_value 1000 的: shape = (1000,20,1)
    #  [[[1.],[2.],...[20.]],[[1.],[2.],...[20.]],[[1.],[2.],...[20.]],...[[1.],[2.],...[20.]]]
    # labels: 大致长度为 max_value 1000 的: shape = (1000,2)
    # [[1.,0.],[1.,0.],[0.,1.],...[1.,0.]]
    # seqlen: 长度为 1000 的: shape = (1000,)
    # [5,13,14,9,11,15,...10,14,17]
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,max_value=1000):
        self.data = []
        self.labels = []
        self.seqlen = []
        self.batch_id = 0

        for i in range(n_samples):

            len = random.randint(min_seq_len, max_seq_len)
            self.seqlen.append(len)

            if random.random() < .5:
                # 小于 0.5 产生线性序列.
                rand_start = random.randint(0, max_value - len)
                s = [[float(i)/max_value] for i in range(rand_start, rand_start + len)]

                # 将所有的序列都扩充至 max_seq_len 最大序列长度. 只有长度相等, 数据集才能被转换成一个 ndarray.
                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)

                # labels 采用 one_hot 编码方式
                self.labels.append([1., 0.])
            else:
                # 大于 0.5 采生随机序列.
                s = [[float(random.randint(0, max_value))/max_value] for i in range(len)]

                s += [[0.] for i in range(max_seq_len - len)]
                self.data.append(s)
                self.labels.append([0., 1.])


    def next(self, batch_size):

        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id + batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        return batch_data, batch_labels, batch_seqlen



import tensorflow as tf


trainset = ToySequenceData(n_samples=1000, max_seq_len=20)
testset = ToySequenceData(n_samples=500, max_seq_len=20)


X = tf.placeholder(dtype=tf.float32, shape=(None,20,1))
Y = tf.placeholder(dtype=tf.float32, shape=(None,2))
seqlen = tf.placeholder(tf.int32, [None])

# X2 为长度为 20 的列表. 数据格式变为: [(None, 1),(None, 1),...(None, 1)]
X2 = tf.unstack(X,20,axis=1)

# lstm_cell 将会输出: (None,64) 的结果. 隐藏层的特征数 n_hidden 指定为 64
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64,forget_bias=1.0)

# 调用 lstm_cell, 输出 20 个 (None, 64) 的 output, outputs 为 [(None,64),(None,64),...(None,64)]. 长度 20.
outputs, state = tf.contrib.rnn.static_rnn(lstm_cell, X2, dtype=tf.float32,sequence_length=seqlen)


# 因为每份样本 X 的长度为 20, 但其后面的一段为 0. 这些数据是没有帮助的.
# 所以我们要取每份样本在 seqlen 处, 也就是样本的最后一个有效值处的 output.
#
# outputs 形状变为: (20,None,64)
outputs = tf.stack(outputs)

# outputs 形状变为: (None, 20, 64)
outputs = tf.transpose(outputs, [1, 0, 2])

# batch_size为:  None
batch_size = tf.shape(outputs)[0]

# 通过 index 索引出每份样本的最后一份有效值.
# 每隔 20 个是每份样本的开始, 加上 seqlen-1 是其最后一个有效值. index 的长度为 None
index = tf.range(0, batch_size) * 20 + (seqlen - 1)

# n_hidden = 64,
# tf.reshape(outputs, [-1, 64]) 输出形状为: (None * 20, 64)
# outputs 变为形状为 (None,64) 的新张量.
outputs = tf.gather(tf.reshape(outputs, [-1, 64]), index)

w1 = tf.Variable(initial_value=tf.random_normal(shape=(64,2),dtype=tf.float32))
b1 = tf.Variable(initial_value=tf.random_normal(shape=(2,),dtype=tf.float32))

logits = tf.matmul(outputs, w1) + b1
pred = tf.math.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,axis=1), tf.argmax(Y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for i in range(10000):
        batch_data, batch_labels, batch_seqlen = trainset.next(128)

        sess.run(optimizer, feed_dict={X:batch_data,Y:batch_labels,seqlen:batch_seqlen})

        if i % 200 == 0 or i == 1:
                    # Calculate batch accuracy & loss
                    accuracy_, cost_ = sess.run([accuracy, cost],
                                                feed_dict={X:batch_data,Y:batch_labels,seqlen:batch_seqlen})

                    print("Step " + str(i) + ", Minibatch Loss= " + \
                          "{:.6f}".format(cost_) + ", Training Accuracy= " + \
                          "{:.5f}".format(accuracy_))
    else:
        print("Optimization Finished!")

        # Calculate accuracy
        test_data = testset.data
        test_label = testset.labels
        test_seqlen = testset.seqlen
        print("Testing Accuracy:",
              sess.run(accuracy, feed_dict={X: test_data, Y: test_label,seqlen: test_seqlen}))
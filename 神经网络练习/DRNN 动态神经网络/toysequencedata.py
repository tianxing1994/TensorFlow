
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

"""
我根据 <OpenCV + TensorFlow 深度学习与计算机视觉实战> 的演示代码, 改写的基于 numpy 的实现.
反馈神经网络原理实现.

输入层, 两个神经元
AX.T + B = Y
X.shape = (1, 3)
A.shape = (6, 3)
B.shape = (6, 1)
Y.shape = (6, 1)
隐藏层, 四个神经元
AX + B = Y
A.shape = (2, 6)
X.shape = (6, 1)
B.shape = (2, 1)
Y.shape = (2, 1)
输出层, 两个神经元(二分类)

矩阵求导: (虽然名为矩阵求导, 但实际只需要注意矩阵运算具体步骤. )
AX = Y
δY / δX = A.T
对 Y 对 X 的导数, 即求, Y 中的对 X 中的每一个值的导数.
示例:
[a1, a2, a3] * [x1, x2, x3].T = y
则:
δy / δx = [δy / δx1, δy / δx2, δy / δx3] = [a1, a2, a3]
δy / δa = [δy / δa1, δy / δa2, δy / δa3] = [x1, x2, x3]
"""
import numpy as np


def sigmoid(x):
    result = 1.0 / (1.0 + np.exp(-x))
    return result


def sigmoid_derivate(x):
    result = sigmoid(x) * (1-sigmoid(x))
    return result


class BPNeuralNetwork(object):
    def __init__(self):
        self.input_array = np.ones(shape=(3, 1), dtype=np.float64)

        self.input_weights = np.random.randn(6, 3)
        self.input_bias = np.random.randn(6, 1)

        self.hidden_array = np.ones(shape=(6, 1), dtype=np.float64)
        self.hidden_array_activated = sigmoid(self.hidden_array)

        self.hidden_weights = np.random.randn(2, 6)
        self.hidden_bias = np.random.randn(2, 1)

        self.output_array = np.ones(shape=(2, 1), dtype=np.float64)
        self.output_array_activated = sigmoid(self.output_array)

    def predict(self, input_array):
        """
        :param input_array: 形状为 (1, 3) 的数组.
        :return: 输出值的形状为: (2, 1)
        """
        self.input_array = input_array.T
        self.hidden_array = np.dot(self.input_weights, self.input_array) + self.input_bias
        self.hidden_array_activated = sigmoid(self.hidden_array)

        self.output_array = np.dot(self.hidden_weights, self.hidden_array_activated) + self.hidden_bias
        self.output_array_activated = sigmoid(self.output_array)
        return self.output_array_activated

    def back_propagate(self, input_array, label, learning_rate):
        """
        :param input_array: 形状为 (1, 3) 的数组
        :param label: 形状为 (1, 2) 的数组
        :param learning_rate:
        :return:
        """
        result = self.predict(input_array)
        label = label.T

        # (2, 1) = (2, 1) - (2, 1)
        output_array_activated_error = label - result

        # 输出层的误差
        # (2, 1) = (2, 1) / (2, 1)
        output_array_error = output_array_activated_error * sigmoid_derivate(self.output_array)

        # (2, 1) / (2, 6) = (2, 6) => sum => (1, 6) => transpose => (6, 1)
        hidden_array_activated_error = np.sum(output_array_error * self.hidden_weights, axis=0, keepdims=True).T

        # 隐藏层的误差
        # (6, 1) / (6, 1) => (6, 1)
        hidden_array_error = hidden_array_activated_error * sigmoid_derivate(self.hidden_array)

        # 梯度下降算法: 求出误差相对于各参数的导数.  learning_rate * f'.
        # 这里是根据各个维度上导数的大小为参考来决定在各个维度上的移动距离.
        # 在本例中, 我们将误差的大小也乘了进来. 这相当于是在误差较大时, 设置更大的学习率.
        # 这里采用 += 运算符, 应注意, 误差的计算是 label - result.
        # 更新 self.hidden_weights
        # (2, 6) += (2, 1) / (1, 6) => (2, 6)
        self.hidden_weights += output_array_error * self.hidden_array.T * learning_rate
        self.hidden_bias += output_array_error * learning_rate

        # 更新 self.input_weights
        # (6, 3) += (6, 1) / (1, 3) => (6, 3)
        self.input_weights += hidden_array_error * self.input_array.T * learning_rate
        self.input_bias += hidden_array_error * learning_rate

        # 计算损失.
        cost = np.sum(np.power(output_array_activated_error, 2)) / output_array_activated_error.shape[0]
        return cost

    def train(self, x_train, y_train, limit=100, learning_rate=0.05):
        for i in range(limit):
            for j in range(len(x_train)):
                input_array = np.reshape(x_train[j], (1, 3))
                label = np.reshape(y_train[j], (1, 2))
                self.back_propagate(input_array, label, learning_rate)
        return


if __name__ == '__main__':
    x_train = np.array([[1, 1, 0],
                        [2, 1, 0],
                        [0, 1, 1],
                        [0, 1, 2]], dtype=np.float64)

    y_train = np.array([[1, 0],
                        [1, 0],
                        [0, 1],
                        [0, 1]], dtype=np.float64)

    nn = BPNeuralNetwork()
    nn.train(x_train, y_train, 10000, 0.005)
    for i in range(len(x_train)):
        input_array = np.reshape(x_train[i], (1, 3))
        label = np.reshape(y_train[i], (2, 1))

        pred = nn.predict(input_array)
        print(pred)
        print(label)


"""
来自 <OpenCV + TensorFlow 深度学习与计算机视觉实战> 的代码演示.
激活函数:
1 / (1 + e^-x)

激活函数的导数:
e^-x / (1 + e^-x)^2


"""
import random
import math


def rand(a, b):
    result = (b - a) * random.random() + a
    return result


def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    """
    1 / (1 + e^-x)
    溢出错误.
    OverflowError: math range error
    I think the value gets too large to fit into a double in python which is why you get the OverflowError.
    The largest value I can compute the exp of on my machine in Python is just sligthly larger than 709.78271.
    https://stackoverflow.com/questions/4050907/python-overflowerror-math-range-error
    https://www.e-learn.cn/index.php/content/wangluowenzhang/186256
    """
    result = 1.0 / (1.0 + math.exp(-x))
    return result


def sigmoid_derivate(x):
    """1 / (1 + e^-x) 的一阶导数, e^-x / (1 + e^-x)^2"""
    result = sigmoid(x) * (1-sigmoid(x))
    return result


class BPNeuralNetwork(object):

    def __init__(self):
        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0
        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []
        self.input_weights = []
        self.output_weights = []

    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no

        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n

        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)

        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)

        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)
        return

    def predict(self, inputs):
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]

        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)

        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)

        result = self.output_cells[:]
        return result

    def back_propagate(self, case, label, learn):
        self.predict(case)

        # 计算输出层的误差
        output_deltas = [0.0] * self.output_n
        for k in range(self.output_n):
            error = label[k] - self.output_cells[k]
            output_deltas[k] = sigmoid_derivate(self.output_cells[k]) * error

        # 计算隐藏层的误差.
        hidden_deltas = [0.0] * self.hidden_n
        for j in range(self.hidden_n):
            error = 0.0
            for k in range(self.output_n):
                error += output_deltas[k] + self.output_weights[j][k]
            hidden_deltas[j] = sigmoid_derivate(self.hidden_cells[j]) * error

        # 更新输出层权重
        for j in range(self.hidden_n):
            for k in range(self.output_n):
                self.output_weights[j][k] += learn * output_deltas[k] * self.hidden_cells[j]

        # 更新隐藏层权重
        for i in range(self.input_n):
            for j in range(self.hidden_n):
                self.input_weights[i][j] += learn * hidden_deltas[j] * self.input_cells[i]

        error = 0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2

        return error

    def train(self, cases, labels, limit=100, learn=0.05):
        for i in range(limit):
            error = 0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn)
        return

    def test(self):
        cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
        labels = [[0], [1], [1], [0]]
        self.setup(2, 5, 1)
        self.train(cases, labels, 10000, 0.05)
        for case in cases:
            print(self.predict(case))
        return


if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()










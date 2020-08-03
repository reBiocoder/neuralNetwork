from data import train_data, test_data

import numpy as np
from numpy.random import randn
import random

# 数据预处理
vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
word_to_ix = {w: i for i, w in enumerate(vocab)}
ix_to_word = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)


def createInputs(text):
    inputs = []
    for w in text.split(" "):
        v = np.zeros((vocab_size, 1))
        v[word_to_ix[w]] = 1
        inputs.append(v)
    return inputs


def softmax(xs):
    return np.exp(xs) / sum(np.exp(xs))


class RNN:
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64):
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs: list):
        # 初始隐藏层输出h为全0
        h = np.zeros((self.Whh.shape[0], 1))
        # 初始化h
        self.last_inputs = inputs
        self.last_hs = {0: h}
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i+1] = h  # 存储每步生成的h的状态

        # 情感分析，此处是many_to_one
        y = self.Why @ h + self.by
        return y, h

    def backprop(self, d_y, learn_rate=2e-2):
        n = len(self.last_inputs)
        # 计算dL/dWhy 和 dL/dWby
        dwhy = d_y @ self.last_hs[n].T
        dby = d_y
        # 计算dL/dWhh,dL/dWxh,dL/dbh的值,方法BPTT算法
        dwhh = np.zeros(self.Whh.shape)
        dwxh = np.zeros(self.Wxh.shape)
        dbh = np.zeros(self.bh.shape)

        # 最后一次的dL/dh
        d_h = self.Why.T @ d_y

        # BPTT
        for t in reversed(range(n)):
            temp = ((1-self.last_hs[t+1] ** 2) * d_h)
            dbh += temp
            dwhh += temp @ self.last_hs[t].T
            dwxh += temp @ self.last_inputs[t].T
            d_h = self.Whh @ temp

        for d in [dwxh,dwhh,dbh,dwhy,dby]:
            np.clip(d, -1, 1, out=d)

        # 梯度下降更新
        self.Whh -= learn_rate * dwhh
        self.Wxh -= learn_rate * dwxh
        self.Why -= learn_rate * dwhy
        self.bh -= learn_rate * dbh
        self.by -= learn_rate *dby


rnn = RNN(vocab_size, 2)


def processData(data: dict, backprop=True):
    items = list(data.items())
    random.shuffle(items)
    loss = 0
    num_correct = 0
    for x, y in items:
        inputs = createInputs(x)  # 特征
        target = int(y)  # 期望输出
        # 前向传播
        out, _ = rnn.forward(inputs)
        probs = softmax(out)
        loss += -np.log(probs[target])  # 期望输出只有两个值，False和True，即为0和1
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            d_L_d_y = probs
            d_L_d_y[target] -= 1
            rnn.backprop(d_L_d_y)
    return loss / len(data), num_correct / len(data)


def predict(text):
    inputs = createInputs(text)
    out, _ = rnn.forward(inputs)
    probs = softmax(out)
    print(np.argmax(probs))


if __name__ == '__main__':
    for epoch in range(1000):
        train_loss, train_acc = processData(train_data)
        if epoch % 100 == 99:
            print('---迭代次数：{}-'.format(epoch+1))
            print("训练：---当前损失：{},当前正确率：{}".format(train_loss, train_acc))

            test_loss, test_acc = processData(test_data, backprop=False)
            print("测试：---当前损失：{},当前正确率：{}".format(test_loss, test_acc))
    print("----------")
    print(predict('this is very bad right now'))
    










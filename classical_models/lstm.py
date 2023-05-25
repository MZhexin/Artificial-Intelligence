# LSTM类 ————> 参考书籍：我妻幸长《写给新手的深度学习2》

import torch
import numpy as np
import matplotlib.pyplot as plt
# import cupy as cp

# 参数设置
n_time = 10                # 时间序列长度
n_in = 1                   # 输入层神经元个数
n_mid = 20                 # 中间层神经元个数
n_out = 1                   # 输出层神经元个数
learning_rate = 0.01       # 学习率
epoch = 50                 # 训练轮数
batch_size = 8             # 批量大小
interval = 10              # 每训练10个批量打印一次结果

# 生成训练数据
sin_x = np.linspace(-2 * np.pi, 2 * np.pi)    # -2π到2π之间的50个数
sin_y = np.sin(sin_x) + 0.1 * np.random.randn(len(sin_x))   # 把sin_x作为sinx的x，做成正弦函数并添加随机噪音
n_sample = len(sin_x) - n_time    # 样本数量————>不理解为什么是这个大小
input_data = np.zeros((n_sample, n_time, n_in))   # 输入数据：一共n_sample个输入数据，每个输入数据都是n_time个时间戳下的n_in个元素
correct_data = np.zeros((n_sample, n_out))   # 正确答案：一共n_sample个正确答案，每个正确答案的大小都是n_out
for i in range(0, n_sample):
    input_data[i] = sin_y[i : i + n_time].reshape(-1, 1)   # 切割输入数据
    correct_data[i] = sin_y[i + n_time : i + n_time + 1]   # 正确答案位于输入数据后一位

# LSTM神经层
class LSTMLayer(object):
    def __init__(self, n_upper, n):
        # LSTM一共有四组需要学习参数
        self.w = np.random.rand(4, n_upper, n) / np.sqrt(n_upper)   # Xavier的初始值
        self.v = np.random.randn(4, n, n) / np.sqrt(n)
        self.b = np.zeros((4, n))

    def forward(self, x, pre_y, pre_c):
        u = np.matmul(x, self.w) + np.matmul(pre_y, self.v) + self.b

        '''
            懒得写sigmoid函数了，直接调用torch
            但因为是学习深度学习的过程，所以不可能直接用torch.lstm()调包的
            嘻嘻
        '''
        a0 = torch.sigmoid(u[0])      # 忘记门
        a1 = torch.sigmoid(u[1])      # 输入门
        a2 = torch.tanh(u[2])         # 输入数据（新的记忆）
        a3 = torch.sigmoid(u[3])      # 输出门

        self.gates = np.stack([a0, a1, a2, a3])   # 堆叠，方便后续调用四个门

        self.c = a0 * pre_c + a1 * a2
        self.y = a3 * torch.tanh(self.c)

    def backward(self, x, y, c, pre_y, pre_c, gates, grad_y, grad_c):
        a0, a1, a2, a3 = gates
        tanh_c = torch.tanh(c)

        # r是人为设置的中间变量，类似delta，为简化后续表达式而存在的
        r = grad_c + grad_y * a3 * (1 - tanh_c ** 2)

        # 各项delta
        delta_a0 = r * pre_c * a0 * (1- a0)
        delta_a1 = r * a2 * a1 * (1 - a1)
        delta_a2 = r * a1 * (1 - a2 ** 2)
        delta_a3 = grad_y * tanh_c * a3 * (1 - a3)

        deltas = np.stack([delta_a0, delta_a1, delta_a2, delta_a3])  # 堆叠，方便调用

        # 各项参数的梯度
        self.grad_w += np.matmul(x.T, deltas)
        self.grad_v += np.matmul(pre_y.T, deltas)
        self.grad_b += np.sum(deltas, axis = 1)

        # x的梯度
        grad_x = np.matmul(deltas, self.w.transpose(0, 2, 1))  # 因为w中包含多个维度，不好直接w.T转置,故用transpose实现转置操作
        self.grad_x = np.sum(grad_x, axis = 0)

        # pre_y的梯度
        grad_pre_y = np.matmul(deltas, self.v.transpose(0, 2, 1))
        self.grad_pre_y = np.sum(grad_pre_y, axis=0)

        # pre_c的梯度
        self.grad_pre_c = r * a0

    # 将累计梯度清零
    def reset_sum_grad(self):
        self.grad_w = np.zeros_like(self.w)
        self.grad_v = np.zeros_like(self.v)
        self.grad_b = np.zeros_like(self.b)

    def update(self, learning_rate):
        self.w -= learning_rate * self.grad_w
        self.v -= learning_rate * self.grad_v
        self.b -= learning_rate * self.grad_b

# 全连接神经层
class OutputLayer(object):
    def __init__(self, n_upper, n):
        self.w = np.random.rand(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros(n)

    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = u

    def backward(self, t):
        delta = self.y - t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)

    def update(self, learning_rate):
        self.w -= learning_rate * self.grad_w
        self.b -= learning_rate * self.grad_b

# 网络各层的初始化
lstm_layer = LSTMLayer(n_in, n_mid)
output_layer = OutputLayer(n_mid, n_out)

# 训练
def train(x_mb, y_mb):
    # 正向传播
    y_lstm = np.zeros((len(x_mb), n_time + 1, n_mid))
    c_lstm = np.zeros((len(x_mb), n_time + 1, n_mid))
# 作业3：感知机（下）
'''
    分别利用随机梯度，最小批次随机梯度和梯度下降对（1）近似后进行优化，并观测优化过程与结果的关系。
'''

# 导库
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# 感知机
class Perceptron:
    def __init__(self, x, y, lr=0.01, epochs=100):
        self.x = x                                                                                                      # 样本
        self.y = y                                                                                                      # 标签
        self.lr = lr                                                                                                    # 学习率（默认0.01）
        self.epochs = epochs                                                                                            # 训练轮数（默认100）

    # 计算单个样本中权重和偏置的梯度
    def gradient(self, x, y, w_grad, b_grad):
        loss = y * (np.dot(w_grad, x) + b_grad)                                                                         # 损失函数 y * (wx + b)
        if loss <= 0:                                                                                                   # 损失小于等于0，即该样本点被错误分类时
            w_grad = x * y                                                                                              # 更新权重的梯度（根据手动计算好的公式）
            b_grad = y                                                                                                  # 更新偏置的梯度
            return w_grad, b_grad                                                                                       # 依次返回权重和偏置的梯度
        else:
            return np.zeros(x.shape), 0                                                                                 # 损失小于等于0，返回0

    # 梯度下降
    def gd(self):
        # 初始化
        samples, features = self.x.shape                                                                                # 样本个数与特征数
        w = np.random.randn(features)                                                                                   # 初始化权重
        b = np.random.randn()                                                                                           # 初始化偏置
        w_grad = np.random.randn(features)                                                                              # 初始化权重的梯度
        b_grad = 0                                                                                                      # 初始化偏置的梯度

        # 训练
        for epoch in tqdm(range(self.epochs)):                                                                          # 每一轮
            for index, item in enumerate(self.x):                                                                       # 每一个样本
                w_grad = self.gradient(item, self.y[index], w_grad, b_grad)[0]                                          # 计算权重的梯度
                b_grad = self.gradient(item, self.y[index], w_grad, b_grad)[1]                                          # 计算偏置的梯度
                w += self.lr * w_grad                                                                                   # 更新权重
                b += self.lr * b_grad                                                                                   # 更新偏置
        return w, b

    # 随机梯度下降
    def sgd(self):
        # 初始化
        samples, features = self.x.shape                                                                                # 样本个数与特征数
        w = np.random.randn(features)                                                                                   # 初始化权重
        b = np.random.randn()                                                                                           # 初始化偏置
        w_grad = np.random.randn(features)                                                                              # 初始化权重的梯度
        b_grad = 0                                                                                                      # 初始化偏置的梯度

        # 训练
        for epoch in tqdm(range(self.epochs)):
            random_index = np.random.randint(len(self.x))                                                               # 设置随机数（随机更新参数的样本索引）
            xi = self.x[random_index: random_index + 1]                                                                 # 随机抽取一个样本点
            yi = self.y[random_index: random_index + 1]
            w_grad = self.gradient(xi[0], yi[0], w_grad, b_grad)[0]                                                     # 计算权重的梯度
            b_grad = self.gradient(xi[0], yi[0], w_grad, b_grad)[1]                                                     # 计算偏置的梯度
            w += self.lr * w_grad                                                                                       # 更新权重
            b += self.lr * b_grad                                                                                       # 更新偏置
        return w, b

    # 小批次梯度下降
    def mbgd(self, batches):
        # 初始化
        samples, features = self.x.shape                                                                                # 样本个数与特征数
        w = np.random.randn(features)                                                                                   # 初始化权重
        b = np.random.randn()                                                                                           # 初始化偏置
        w_grad = np.random.randn(features)                                                                              # 初始化权重的梯度
        b_grad = 0                                                                                                      # 初始化偏置的梯度

        # 训练
        for epoch in tqdm(range(self.epochs)):
            random_index = np.random.randint(len(self.x) - batches)                                                     # 设置随机数（减去批量大小以免超出索引范围）
            xi = self.x[random_index: random_index + batches]                                                           # 随机抽取batches个样本点
            yi = self.y[random_index: random_index + batches]
            for batch in range(batches):
                w_grad = self.gradient(xi[batch], yi[batch], w_grad, b_grad)[0]                                         # 计算权重的梯度
                b_grad = self.gradient(xi[batch], yi[batch], w_grad, b_grad)[1]                                         # 计算偏置的梯度
                w += self.lr * w_grad                                                                                   # 更新权重
                b += self.lr * b_grad                                                                                   # 更新偏置
        return w, b

# 创建数据集（这部分借鉴班里同学的代码）
# 定义多元高斯分布的均值和协方差矩阵
mean_pos = np.zeros(5)                                                                                                  # 正面样本的均值
mean_neg = np.ones(5)                                                                                                   # 负面样本的均值
cov = np.eye(5)                                                                                                         # 协方差矩阵

# 生成正样本和负样本
np.random.seed(0)                                                                                                       # 设置随机数生成器种子以确保结果的可重复性
n_samples = 400                                                                                                         # 每类样本的数量

pos_samples = np.random.multivariate_normal(mean_pos, cov, n_samples)
neg_samples = np.random.multivariate_normal(mean_neg, cov, n_samples)

# 将样本数据和标签合并为一个数组
x = np.vstack([pos_samples, neg_samples])
y = np.hstack([np.ones(n_samples), -np.ones(n_samples)])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 实例化
model = Perceptron(x_train, y_train, lr=0.01, epochs=100)

# 计算权重与偏置
w_gd, b_gd = model.gd()                                                                                                 # 梯度下降
w_sgd, b_sgd = model.sgd()                                                                                              # 随机梯度下降
w_mbgd, b_mbgd = model.mbgd(batches=20)                                                                                 # 小批次梯度下降

# 用测试集预测，计算损失函数（Loss > 0 视为预测正确）
loss_gd = (np.dot(x_test, w_gd) + b_gd) * y_test                                                                        # 梯度下降
loss_sgd = (np.dot(x_test, w_sgd) + b_sgd) * y_test                                                                     # 随机梯度下降
loss_mbgd = (np.dot(x_test, w_mbgd) + b_mbgd) * y_test                                                                  # 小批次梯度下降

# 计算准确率
correct_gd, correct_sgd, correct_mbgd = 0, 0, 0                                                                         # 初始化正确个数
for index, item in enumerate(y_test):
    if loss_gd[index] > 0:                                                                                              # 梯度下降
        correct_gd += 1
    if loss_sgd[index] > 0:                                                                                             # 随机梯度下降
        correct_sgd += 1
    if loss_mbgd[index] > 0:                                                                                            # 小批次梯度下降
        correct_mbgd += 1

# 打印准确率
print('ACC of GD is: {0}'.format(correct_gd / len(y_test)))
print('ACC of SGD is: {0}'.format(correct_sgd / len(y_test)))
print('ACC of MBGD is: {0}'.format(correct_mbgd / len(y_test)))
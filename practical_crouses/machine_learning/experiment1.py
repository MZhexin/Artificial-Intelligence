# 机器学习基础实验课程 实验一
'''
    1. 实验要求-基本分项
        列出回归方程，根据实验数据，随机选取训练集和测试集；
        分别利用最小二乘法、梯度下降法等方法进行求解；
        画出散点图，计算测试集的平方和误差、均方根误差、平均绝对误差，分析两种解法的优劣。
    2. 实验要求-加分项
        可自行寻找实验数据，进行扩展实验；
        进行多元线性回归分析。
'''

# 导库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 梯度
def gradient(x, y, w):
    gradient = np.dot(x.T, np.dot(w.T, x) - y)
    return gradient

# 最小二乘法
def LSM(x, y):
    one = np.ones(len(x))
    x_b = np.c_[one, x]
    pinv = np.linalg.pinv(np.dot(x_b.T, x_b))
    w = np.dot(np.dot(pinv, x_b.T), y)
    return w

# 梯度下降法
def GD(x, y, learning_rate, epochs):
    m = len(x)
    one = np.ones(len(x))
    x_b = np.c_[one, x]
    w = np.zeros(2)
    for epoch in range(epochs):
        w -= 2 * learning_rate * np.dot(np.dot(w, x_b.T) - y, x_b) / m
    return w

# 误差指标
def analyze_model(y_true, y_pred, method_name):
    sse = np.sum((y_true - y_pred) ** 2)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    print("{0}:\n \tSSE: {1}\n \tRMSE: {2}\n \tMAE: {3}".format(method_name, sse, rmse, mae))

# 导入数据集
data = pd.read_csv('dataset/SD.csv', header=0)
cost, price = np.array(data['成本']), np.array(data['价格'])

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(cost, price, test_size=0.2, random_state=22)

# 代入训练集，求解
w_lsm = LSM(x_train, y_train)
w_GD = GD(x_train, y_train, learning_rate=0.01, epochs=50000)

# 代入测试集，预测
# 增广
one = np.ones(len(x_test))
x_b = np.c_[one, x_test]
y_predict_lsm = np.dot(w_lsm, x_b.T)
y_predict_GD = np.dot(w_GD, x_b.T)

# 评价
analyze_model(y_test, y_predict_lsm, 'LSM')
analyze_model(y_test, y_predict_GD, 'GD')

# 绘图
figure, ax = plt.subplots()
ax.plot(x_test, y_predict_lsm, color='b', label='LSM')
ax.plot(x_test, y_predict_GD, color='r', label='GD')
ax.scatter(cost, price, color='g', marker='o', label='Points')
ax.legend()
ax.set_xlabel('Cost', fontsize=12)
ax.set_ylabel('Price', fontsize=12)
ax.set_title('Experiment 1')
ax.tick_params(axis='both', which='major', labelsize=10)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_facecolor('#f0f0f0')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
plt.show()
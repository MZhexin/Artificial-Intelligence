# 机器学习基础实验课程 实验二
'''
    实验内容：
        列出回归方程，根据实验数据进行训练；
        分别利用梯度下降法、牛顿法等方法进行求解；
        画出散点图，计算准确率，分析两种解法的优劣。
'''

# 导库
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

def data_load():
    # 读取训练集和测试集txt文件
    train_data = pd.read_csv('dataset/horseColicTraining.txt', sep='\t')
    test_data = pd.read_csv('dataset/horseColicTest.txt', sep='\t')

    # 添加列名
    column_names = [
        "手术", "年龄", "体温", "脉搏", "呼吸频率", "温度的极端性", "脉搏的极端性", "粘膜颜色",
        "毛细血管充盈时间", "疼痛", "蠕动", "腹部肿胀", "鼻饲", "肠蠕动声", "腹痛",
        "腹腔穿刺液体", "腹部的问题", "白细胞计数", "血红蛋白量", "输血前的包细胞体积百分比", "总蛋白质", "结果"
    ]

    train_data.columns = column_names
    test_data.columns = column_names

    # 将手术特征转换为二元变量
    train_data['手术'] = train_data['手术'].apply(lambda x: 1 if x == 1 else 0)
    test_data['手术'] = test_data['手术'].apply(lambda x: 1 if x == 1 else 0)

    # 提取特征和标签
    x_train = train_data.drop('结果', axis=1)
    y_train = train_data['结果']
    x_test = test_data.drop('结果', axis=1)
    y_test = test_data['结果']

    # 增广
    one_train, one_test = np.ones(len(x_train)), np.ones(len(x_test))
    x_train = np.c_[one_train, x_train]
    x_test = np.c_[one_test, x_test]

    return x_train, y_train, x_test, y_test

# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 梯度下降法
def gd(x, y, lr=0.001, epochs=1000):
    m = len(y)
    w = np.zeros(x.shape[1])
    for epoch in tqdm(range(epochs)):
        h = sigmoid(np.dot(x, w))
        gradient = np.dot(x.T, (h - y)) / m
        w -= lr * gradient
    return w

# 牛顿法
def newton(x, y, epochs=10):
    m = len(y)
    w = np.zeros(x.shape[1])
    for epoch in tqdm(range(epochs)):
        h = sigmoid(np.dot(x, w))
        gradient = np.dot(x.T, (h - y)) / m
        Hessian = np.dot(x.T, np.dot(np.diag(h * (1 - h)), x)) / m
        w -= np.dot(np.linalg.inv(Hessian), gradient)
    return w

# 准确率
def accuracy(y_true, y_pred):
    y_pred_binary = np.round(y_pred)
    return np.mean(y_true == y_pred_binary)

# 主函数
def main():
    # 读取数据
    x_train, y_train, x_test, y_test = data_load()

    w_gd = gd(x_train, y_train, lr=0.001, epochs=1000)
    w_newton = newton(x_train, y_train, epochs=10)

    y_predict_gd = sigmoid(np.dot(x_test, w_gd))
    y_predict_newton = sigmoid(np.dot(x_test, w_newton))

    # 计算准确率
    acc_gd = accuracy(y_test, y_predict_gd)
    acc_newton = accuracy(y_test, y_predict_newton)

    print('ACC of GD: {0}'.format(acc_gd))
    print('ACC of Newton: {0}'.format(acc_newton))

    # 绘图
    figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # 梯度下降法
    ax[0].scatter(y_predict_gd, y_test, color='b', label='GD')
    ax[0].legend()
    ax[0].set_title('Predict-Real Scatter Plot(Gradient Descent Method)', fontsize=14)
    ax[0].tick_params(axis='both', which='major', labelsize=10)
    ax[0].grid(True, linestyle='--', alpha=0.7)
    ax[0].set_facecolor('#f0f0f0')
    ax[0].spines['top'].set_color('None')
    ax[0].spines['right'].set_color('None')

    # 牛顿法
    ax[1].scatter(y_predict_newton, y_test, color='r', label='Newton')
    ax[1].legend()
    ax[1].set_title('Predict-Real Scatter Plot(Newton Method)', fontsize=14)
    ax[1].tick_params(axis='both', which='major', labelsize=10)
    ax[1].grid(True, linestyle='--', alpha=0.7)
    ax[1].set_facecolor('#f0f0f0')
    ax[1].spines['top'].set_color('None')
    ax[1].spines['right'].set_color('None')

    # 显示图像
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
# 作业2：梯度下降
# 给定函数和初值，利用梯度下降公式寻找最优解并观察不同学习率对结果的影响
# 要求：手写梯度下降法，不允许调库

# 导库
import numpy as np

# 原函数
def function(x, y):
    # 原函数为x - y的L2范数的平方
    return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2

# 原函数的导数
def derivative(x, y):
    result = np.zeros(2)
    for i in range(len(result)):
        result[i] = 2 * (x[i] - y[i])
    return result

# 初始化参数
x0 = np.zeros(2)                                # 初始点
y = np.array([3.3, 2.001])                      # 参数
lr = 0.5                                        # 学习率

# 梯度下降法
def GD(x0, y, lr, epoch=50):                    # 固定训练轮数为50
    for i in range(epoch):
        x0 = x0 - lr * derivative(x0, y)        # 更新梯度
    # 打印结果
    print('Minimum Point: {0}, Function Value: {1}'.format(x0, function(x0, y)))

# 执行
GD(x0, y, lr)
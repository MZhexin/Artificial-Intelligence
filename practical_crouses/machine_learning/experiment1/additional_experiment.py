# 机器学习基础实验课程 实验一
'''
    附加实验内容：
        可自行寻找实验数据，进行扩展实验；
        进行多元线性回归分析。
'''

# 导库
import random
import numpy as np
import matplotlib.pyplot as plt

# 创建数据集
x1 = np.linspace(0, 60, 100) + 5 * np.random.normal(size=100)
x2 = np.linspace(0, 65, 100) + 5 * np.random.normal(size=100)
y = np.linspace(0, 70, 100) + 5 * np.random.normal(size=100)

# 划分训练集和测试集
x1_train, x2_train, y_train = np.sort(np.array(random.sample(list(x1), 80))), np.zeros(80), np.zeros(80)                # 训练集：随机抽取80个x1，初始化x2和y
x1_test, x2_test, y_test = np.sort(np.array(random.sample(list(x1), 20))), np.zeros(20), np.zeros(20)                   # 测试集：随机抽取80个x1，初始化x2和y

# 找到对应索引，完成x2和y的提取
for index, data in enumerate(x1):
    for idx, item in enumerate(x1_train):
        if x1_train[idx] == x1[index]:
            x2_train[idx] = x2[index]
            y_train[idx] = y[index]
for index, data in enumerate(x1):
    for idx, item in enumerate(x1_test):
        if x1_test[idx] == x1[index]:
            x2_test[idx] = x2[index]
            y_test[idx] = y[index]

# 用最小二乘法求解多元线性回归（增广、求伪逆、计算w）
one = np.ones(80)
train_data = np.c_[one, x1_train]
train_data = np.c_[train_data, x2_train]
pinv = np.linalg.pinv(np.dot(train_data.T, train_data))
w = np.dot(np.dot(pinv, train_data.T), y_train)

# 预测（先增广）
one = np.ones(20)
test_data = np.c_[one, x1_test]
test_data = np.c_[test_data, x2_test]
y_predict = np.dot(w, test_data.T)

# 绘图
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, color='r', marker='o', label='Points', alpha=0.8)
ax.plot(x1_test, x2_test, y_predict, color='b', label='Line')
ax.legend(loc='best')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Multiple Linear Regression Analysis', fontsize=16, fontweight='bold')
ax.set_xlim(0, 80)
ax.set_ylim(0, 80)
ax.set_zlim(0, 80)
ax.view_init(elev=20, azim=30)
plt.show()
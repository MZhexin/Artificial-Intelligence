# 导库
import numpy as np
import matplotlib.pyplot as plt

# 直方图（Histogram）
# 参考教程：https://www.bilibili.com/video/BV1wP411k7NM?p=4
'''
    （1）特点：数据的分布情况，用矩形条表示数据频率
    （2）用途：分析数据的分布和形状，检测数据的偏态和峰值
'''

# 生成显示数据：服从标准正态分布的100个点
np.random.seed(0)
data1 = np.random.normal(0, 1, 100)

# 创建Figure对象和子图
fig, ax = plt.subplots()

# 设置直方图的边界和颜色
bins = 20
colors = ['g']

# 绘制直方图
ax.hist(data1, bins=bins, color=colors, alpha=0.7, label='Group', edgecolor='red')

# 添加图例
ax.legend()

#添加横纵坐标标签
ax.set_xlabel('Data', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)

# 设置图表标题
ax.set_title('Histogram', fontsize=14)

# 设置刻度标签字体大小
ax.tick_params(axis='both', which='major', labelsize=10)

# 设置背景颜色
ax.set_facecolor('#f0f0f0')

# 设置图表边框颜色
ax.spines['top'].set_color('None')
ax.spines['right'].set_color('None')

# 显示图像
plt.show()
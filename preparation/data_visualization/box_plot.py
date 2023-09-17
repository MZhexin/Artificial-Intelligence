# 导库
import numpy as np
import matplotlib.pyplot as plt

# 箱线图（Box Plot）
# 参考教程 1：https://www.bilibili.com/video/BV1wP411k7NM?p=7
# 参考教程 2：http://t.csdn.cn/HcjFR
'''
    （1）特点：显示数据的统计分布情况，包括中位数、四分位数等
    （2）用途：检测数据的离群点和异常值，比较不同组数据的分布，美观且高效
'''

# 生成示例数据
np.random.seed(0)
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(2, 1.5, 100)
data3 = np.random.normal(-2, 1.5, 100)
data4 = np.random.normal(3, 1, 100)
data5 = np.random.normal(-1, 0.5, 100)

# 创建Figure对象和子图
fig, ax = plt.subplots()

# 绘制箱线图
boxes = ax.boxplot([data1, data2, data3, data4, data5],                                # 数据
                    labels=['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5'],     # 标签
                    sym='o',                                                            # 异常点的形状
                    vert=True,                                                          # 是否垂直摆放
                    patch_artist=True)                                                  # 是否填充箱体颜色

# 设置每组数据的颜色，箱线图需要在boxplot函数外设置颜色
colors = ['tab:blue', 'yellow', 'tab:green', 'tab:red', 'tab:purple']
for box, color in zip(boxes['boxes'], colors):
    box.set(facecolor=color)

# 添加图例
ax.legend(boxes['boxes'], ['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5'], loc='best')

# 设置图表标题
ax.set_title('Box Plot', fontsize=14)

# 设置刻度标签字体大小
ax.tick_params(axis='both', which='major', labelsize=10)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.7)

# 设置背景颜色
ax.set_facecolor('#f0f0f0')

# 设置图表边框颜色
ax.spines['top'].set_color('None')
ax.spines['right'].set_color('None')

# 显示图像
plt.tight_layout()  # 作用：自动调整子图参数，使之填充整个图像区域
plt.show()
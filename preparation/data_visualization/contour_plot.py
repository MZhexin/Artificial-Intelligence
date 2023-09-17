# 导库
import numpy as np
import matplotlib.pyplot as plt

# 等高线图（Contour Plot）
'''
    （1）特点：二维数据的等高线图，用线表示等高线
    （2）用途：可视化函数的等高线，显示函数的高低和形状
'''

# 生成示例数据
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x, y)        # 根据输入的坐标向量生成对应的坐标矩阵
Z = X ** 2 - Y ** 2             # 等高线图除了XY坐标外，还需要第三个维度Z，即高度信息

# 创建Figure对象和子图
fig, ax = plt.subplots()

# 绘制等高线图
contour = ax.contour(X, Y, Z,           # 三个维度的坐标
                     levels=10,         # levels越小，等高线越稀疏
                     cmap='coolwarm',   # 颜色设置
                     linewidths=1.5)    # 线条宽度

# 添加颜色条
cbar = plt.colorbar(contour, ax=ax)

# 设置图表标题
ax.set_title('contour Plot', fontsize=16, fontweight='bold')

# 添加横纵坐标的标签
ax.set_xlabel('X Axis', fontsize=14, fontweight='bold')
ax.set_ylabel('Y Axis', fontsize=14, fontweight='bold')

# 设置颜色条标签字体大小
cbar.ax.tick_params(labelsize=12)

# 设置刻度标签大小
ax.tick_params(axis='both', which='major', labelsize=10)

# 设置线条样式为虚线
for line in contour.collections:
    line.set_linestyle('dashed')

# 设置背景颜色
ax.set_facecolor('#f0f0f0')

# 设置图表边框颜色
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# 显示图像
plt.tight_layout()
plt.show()
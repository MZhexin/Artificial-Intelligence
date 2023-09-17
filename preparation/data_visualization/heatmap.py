# 导库
import numpy as np
import matplotlib.pyplot as plt

# 热力图（Heatmap）
'''
    （1）特点：二维数据的颜色编码图，用颜色表示数值大小
    （2）用途：可视化矩阵或二维数组，观察数据的相关性和关联程度
'''

# 生成示例数据（10×10矩阵）
data = np.random.randint(1, 10, size=(10, 10))

# 创建Figure对象和子图
fig, ax = plt.subplots()

# 绘制热力图
heatmap = ax.imshow(data, cmap='hot', interpolation='nearest', aspect='auto')

# 添加颜色条
cbar = plt.colorbar(heatmap, fraction=0.046, pad=0.04)

# 添加横纵坐标的标签
ax.set_xlabel('X Axis', fontsize=14)
ax.set_ylabel('Y Axis', fontsize=14)

# 设置图表标题
ax.set_title('Heatmap', fontsize=16)

# 设置颜色条标签字体大小
cbar.ax.tick_params(labelsize=10)

# 设置刻度标签大小
ax.tick_params(axis='both', which='major', labelsize=10)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.7)

# 设置背景颜色
ax.set_facecolor('#f0f0f0')

# 设置图表边框颜色
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')

# 显示图像
plt.tight_layout()
plt.show()
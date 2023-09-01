# 导库
import numpy as np
import matplotlib.pyplot as plt

# 3D图（3D Plot）
# 展示3D函数图像、3D散点图和3D柱状图三种3D图像的绘制方法
'''
    （1）特点：三维数据的可视化图像，如散点图、曲面图、柱状图等
    （2）用途：展示三维数据的关系和分布，可视化空间数据
'''

# 1. 3D函数图像（3D Stereo Function Image）
# 定义函数
def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

# 生成网格点
x = np.linspace(-5, 5, 100)     # x序列
y = np.linspace(-5, 5, 100)     # y序列
X, Y = np.meshgrid(x, y)        # 将x和y张成一个二维平面
# print(X)                      # 测试结果为100×100的数组，其中每行都是一维数组x，行数与数组y的长度相同
Z = f(X, Y)

# 创建3D图形对象
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制立体函数图像
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='black', alpha=0.8)

# 设置坐标轴标签
ax.set_label('X')
ax.set_label('Y')
ax.set_label('Z')

# 设置坐标轴范围
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-1, 1)

# 设置刻度标签
ax.set_xticks(np.arange(-5, 6, 2))
ax.set_yticks(np.arange(-5, 6, 2))
ax.set_zticks(np.arange(-1, 1.5, 0.5))

# 设置标题
plt.title('3D Stereo Function Image')

# 设置颜色条
cbar = plt.colorbar(surf, shrink=0.5, aspect=5)

# 旋转视角、调整角度
ax.view_init(elev=30, azim=30)

# 隐藏边框
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('w')
ax.yaxis.pane.set_edgecolor('w')
ax.zaxis.pane.set_edgecolor('w')
ax.grid(True)

# 显示图像
plt.show()


# 2. 3D散点图
# 生成示例数据
np.random.seed(0)
cluster1 = np.random.randn(100, 3) + [3, 3, 3]
cluster2 = np.random.randn(100, 3) + [-2, -2, -2]
cluster3 = np.random.randn(100, 3) + [1, -1, 4]

# 创建3D图形对象
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维数据点
ax.scatter(cluster1[:, 0], cluster1[:, 1], cluster1[:, 2], color='r', marker='o', label='Cluster 1', alpha=0.8)
ax.scatter(cluster2[:, 0], cluster2[:, 1], cluster2[:, 2], c='g', marker='^', label='Cluster 2', alpha=0.8)
ax.scatter(cluster3[:, 0], cluster3[:, 1], cluster3[:, 2], c='b', marker='s', label='Cluster 3', alpha=0.8)

# 设置图例
ax.legend(loc='upper right')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置标题
plt.title('3D Scatter Plot', fontsize=16, fontweight='bold')

# 设置坐标轴刻度范围
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_zlim(-6, 6)

# 调整视角
ax.view_init(elev=20, azim=30)

# 显示图像
plt.show()


# 3. 3D柱状图（3D Bar Chart）
# 生成示例数据
x = [1, 2, 3]
y = [1, 2, 3, 4, 5]
z = [[5, 4, 2],
     [7, 6, 3],
     [7, 5, 4],
     [6, 7, 3],
     [5, 6, 2]]

# 创建3D图形对象
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制三维柱状图
dx = dy = 0.5               # 设置柱子宽度
dz = [row[0] for row in z]  # 设置柱子高度

color = ['tab:red', 'y', 'tab:green']
for i in range(len(x)):
    for j in range(len(y)):
        ax.bar3d(x[i], y[j], 0, dx, dy, dz[j], shade=True, color=color[i], edgecolor='black',
                 linewidth=1, alpha=0.7)

# 创建虚拟的图例
rect1 = plt.Rectangle((0, 0), 1, 1, fc=color[0], edgecolor='black', linewidth=1)
rect2 = plt.Rectangle((0, 0), 1, 1, fc=color[1], edgecolor='black', linewidth=1)
rect3 = plt.Rectangle((0, 0), 1, 1, fc=color[2], edgecolor='black', linewidth=1)
ax.legend([rect1, rect2, rect3], ['Category 1', 'Category 2', 'Category 3'], loc='upper left')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 设置标题
plt.title('3D Bar Chart')

# 显示图像
plt.show()
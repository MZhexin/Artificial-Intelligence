# 导库
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 散点图（Scatter Plot）
# 参考教程：https://www.bilibili.com/video/BV1wP411k7NM?p=5
'''
    （1）特点：二维数据点的分布，点在二维平面上
    （2）用途：观察两个变量之间的关系，检测数据的聚集和离群点
'''

# 生成示例数据
np.random.seed(0)
data = np.random.rand(100, 2) * 5                       # 100个随机数据点，取值范围在[0, 5)

# 使用K-Means进行聚类
num_cluster = 3
kmeans = KMeans(n_clusters=num_cluster, random_state=0)
labels = kmeans.fit_predict(data)

# 创建Figure对象和子图
fig, ax = plt.subplots()

# 设置颜色列表和标记样式列表
colors = ['tab:blue', 'tab:orange', 'tab:green']        # tab是不同的颜色表示方式，tab:blue比blue要淡一些，更好看
markers = ['o', 's', 'D']                               # o、s、D依次表示圆形、方形和菱形

# 绘制散点图
for i in range(num_cluster):
    cluster_data = data[labels == i]                    # 把所有类别是i的点取出来
    ax.scatter(cluster_data[:, 0],                      # X坐标
               cluster_data[:, 1],                      # Y坐标
               marker=markers[i],                       # 标志
               color=colors[i],                         # 颜色
               label='Cluster {0}'.format(i + 1))       # 标签

# 添加图例
ax.legend()

# 设置图表标题
ax.set_title('Scatter Plot', fontsize=14)

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
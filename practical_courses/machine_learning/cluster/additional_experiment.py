# 机器学习基础实验课程 实验三
'''
    实验要求-加分项
        在数据集上对比不同聚类方法、不同评价指标的实验结果；
        可自行寻找实验数据，进行扩展实验。
'''

'''
    注：此代码全部来自本人的舍友
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

# 加载 Wine 数据集
wine = load_wine()
X = wine.data

# 使用Ward linkage进行层次聚类
linkage_matrix = linkage(X, 'ward')

# 绘制树状图（树状图可以帮助我们确定合适的簇数）
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, truncate_mode='level', p=3)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# 根据树状图确定簇的数量，然后进行聚类
num_clusters = 3  # 根据树状图确定簇的数量
labels = np.array([i for i in range(len(X))])  # 初始化每个样本为一个簇

# 切割树状图以获得簇
for _ in range(len(X) - num_clusters):
    merge = linkage_matrix[0]
    labels[labels == merge[0]] = len(X) + _
    labels[labels == merge[1]] = len(X) + _
    linkage_matrix = linkage_matrix[1:]

# 打印最终簇的分配情况
print('Final Clustering Labels:', labels)

# 计算轮廓系数
silhouette_avg = silhouette_score(X, labels)
print(f'Silhouette Score: {silhouette_avg}')

# 绘制层次聚类的散点图
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolors='k', s=50, alpha=0.7)
plt.title('Hierarchical Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

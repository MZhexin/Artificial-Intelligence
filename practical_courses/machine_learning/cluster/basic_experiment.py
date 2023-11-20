# 机器学习基础实验课程 实验三
'''
    实验要求-基本分项
        根据实验数据、利用K-均值进行聚类；
        分别利用不同距离度量进行聚类；
        列出迭代过程中每类样本的变化情况，计算轮廓系数，分析两种度量下聚类效果的优劣。

'''

'''
    参考资料：
        http://t.csdnimg.cn/xCP9B
        http://t.csdnimg.cn/a8Pgu
        https://zhuanlan.zhihu.com/p/122195108
'''

# 导库
import warnings
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

warnings.filterwarnings("ignore")                                                                                       # 忽视警告（无视风险，继续安装.jpg）

# KMeans
class KMeans():
    def __init__(self):
        pass

    # 计算距离
    def calc_distance(self, diff, distance_type):
        # 欧式距离
        if distance_type == 'euclidean':
            return np.linalg.norm(diff, axis=2)
        elif distance_type == 'manhattan':
            return np.sum(np.abs(diff), axis=2)

    def fit(self, x, k=3, epochs=20, distance_type='euclidean'):
        # 初始化
        n_samples, n_features = x.shape
        centroids = x[np.random.choice(n_samples, k, replace=False)]

        for epoch in tqdm(range(epochs)):
            distances = self.calc_distance(x[:, np.newaxis, :] - centroids, distance_type=distance_type)                # 计算距离
            labels = np.argmin(distances, axis=1)                                                                       # 分配到最近的聚类中心
            new_centroids = np.array([x[labels == j].mean(axis=0) for j in range(k)])                                   # 计算新的聚类中心
            centroids = new_centroids                                                                                   # 更新聚类中心

        return centroids, labels

# 主函数
def main():
    iris = load_iris()                                                                                                  # 加载鸢尾花数据集
    X = iris.data[:, 2:]                                                                                                # 选择后两个特征
    model = KMeans()                                                                                                    # 实例化模型
    centroids, labels = model.fit(X, k=3, epochs=20, distance_type='euclidean')                                         # 聚类

    # 绘图
    figure, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=labels, marker='o', s=50, label='Data Points')
    ax.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='s', s=100, label='Centroids')
    ax.set_title('K-Means Clustering', fontsize=14)
    ax.set_xlabel('Feature 3')
    ax.set_ylabel('Feature 4')
    ax.legend()
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('#f0f0f0')
    ax.spines['top'].set_color('None')
    ax.spines['right'].set_color('None')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
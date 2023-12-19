# 机器学习基础实验课程 实验七
'''
    基础实验内容：
        利用主成分分析方法对实验数据进行降维；
        结合K-means聚类方法，对降维后的实验数据进行聚类；
        选取不同的降维数量，计算准确率，画出维度-准确率曲线。
    附加实验内容：
        任选其它两种分类或聚类算法，做出相应的降维实验结果；
        可自行寻找实验数据，进行扩展实验。
'''

'''
    由于聚类不存在准确率的说法，因此用其他指标代替————> e.g. 轮廓系数
'''

# 导库
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering

# PCA
def pca(X, n_components):
    # 中心化数据
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # 计算协方差矩阵
    cov_matrix = np.cov(X_centered, rowvar=False)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 选择前n个特征向量
    top_eigenvectors = eigenvectors[:, -n_components:]

    # 投影数据到新的空间
    X_pca = np.dot(X_centered, top_eigenvectors)

    return X_pca

# 读取数据集
data = pd.read_csv('dataset/yaleB.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# 初始化维度和轮廓系数的列表（K均值、K近邻和层次聚类）
dimensions_KMeans = []
dimensions_KNN = []
dimensions_AGG = []
silhouette_scores_KMeans = []
silhouette_scores_KNN = []
silhouette_scores_AGG = []

# K均值
for n_components in tqdm(range(1, X.shape[1] + 1, 20)):
    # PCA降维
    X_pca = pca(X, n_components)

    # K-means
    kmeans = KMeans(n_clusters=3, n_init=10)
    y_pred = kmeans.fit_predict(X_pca)

    # 轮廓系数
    silhouette_score = metrics.silhouette_score(X_pca, y_pred)

    # 保存维度和轮廓系数
    dimensions_KMeans.append(n_components)
    silhouette_scores_KMeans.append(silhouette_score)

# K近邻
for n_components in tqdm(range(1, X.shape[1] + 1, 20)):
    # PCA降维
    X_pca = pca(X, n_components)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    y_pred = knn.fit(X_pca, y).predict(X_pca)

    # 轮廓系数
    silhouette_score = metrics.silhouette_score(X_pca, y_pred)

    # 保存维度和轮廓系数
    dimensions_KNN.append(n_components)
    silhouette_scores_KNN.append(silhouette_score)

# 层次聚类
for n_components in tqdm(range(1, X.shape[1] + 1, 20)):
    # PCA降维
    X_pca = pca(X, n_components)

    # 层次聚类
    agg_clustering = AgglomerativeClustering(n_clusters=3)
    y_pred = agg_clustering.fit_predict(X_pca)

    # 轮廓系数
    silhouette_score = metrics.silhouette_score(X_pca, y_pred)

    # 保存维度和轮廓系数
    dimensions_AGG.append(n_components)
    silhouette_scores_AGG.append(silhouette_score)

# 绘制维度-轮廓系数曲线
plt.plot(dimensions_KMeans, silhouette_scores_KMeans, marker='o')
plt.title('PCA With KMeans')
plt.xlabel('Number of Dimensions')
plt.ylabel('Silhouette Score')
plt.show()

# 绘制维度-轮廓系数曲线
plt.plot(dimensions_KNN, silhouette_scores_KNN, marker='s')
plt.title('PCA With KNN')
plt.xlabel('Number of Dimensions')
plt.ylabel('Silhouette Score')
plt.show()

# 绘制维度-轮廓系数曲线
plt.plot(dimensions_AGG, silhouette_scores_AGG, marker='d')
plt.title('PCA With Agglomerative Clustering')
plt.xlabel('Number of Dimensions')
plt.ylabel('Silhouette Score')
plt.show()
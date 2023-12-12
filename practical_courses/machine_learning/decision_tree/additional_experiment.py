# 机器学习基础实验课程 实验六
'''
    附加实验内容：
        在C4.5决策树模型的基础上，分别加入预剪枝和后剪枝过程，对比实验结果；
        可自行寻找实验数据，进行扩展实验。
'''

# 导库
import numpy as np
import pandas as pd

class C45():
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth                                                                                      # 最大深度限制
        self.min_samples_split = min_samples_split                                                                      # 决定是否分裂一个节点所需要的最小样本数
        self.min_samples_leaf = min_samples_leaf                                                                        # 叶子节点所需要的最小样本数

    def fit(self, X, y):
        self.tree = self._create_tree(X, y, list(range(X.shape[1])), depth=0)

    # 计算熵
    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)                                                              # 获取y中所有的唯一类和对应的计数
        probabilities = counts / len(y)                                                                                 # 计算每个类的概率
        entropy = -np.sum(probabilities * np.log2(probabilities))                                                       # 计算熵
        return entropy

    # 计算信息增益比
    def _information_gain_ratio(self, X, y, feature_idx):
        entropy_parent = self._entropy(y)                                                                               # 计算父节点的熵值

        classes, counts = np.unique(X[:, feature_idx], return_counts=True)                                              # 获取特征的唯一值和对应的计数
        weighted_entropy = 0                                                                                            # 初始化加权熵值为0
        split_info = 0                                                                                                  # 初始化分裂信息为0
        for c in classes:                                                                                               # 对于特征的每一个唯一值
            subset_y = y[X[:, feature_idx] == c]                                                                        # 获取该唯一值对应的子集的目标变量
            subset_prob = len(subset_y) / len(y)                                                                        # 计算子集的权重（子集大小/总集大小）
            weighted_entropy += subset_prob * self._entropy(subset_y)                                                   # 计算加权熵值
            split_info -= subset_prob * np.log2(subset_prob)                                                            # 计算分裂信息

        information_gain = entropy_parent - weighted_entropy                                                            # 计算信息增益
        information_gain_ratio = information_gain / split_info                                                          # 计算信息增益比
        return information_gain_ratio

    # 创建决策树：包含剪枝过程
    def _create_tree(self, X, y, features, depth):
        if len(np.unique(y)) == 1:
            return y[0]
        elif depth == self.max_depth:
            return np.bincount(y).argmax()
        elif len(y) < self.min_samples_split:
            return np.bincount(y).argmax()
        elif len(features) == 0:
            return np.bincount(y).argmax()
        else:
            best_feature = max(features, key=lambda i: self._information_gain_ratio(X, y, i))
            tree = {best_feature: {}}
            remaining_features = [f for f in features if f != best_feature]
            for value in np.unique(X[:, best_feature]):
                subset_indices = np.where(X[:, best_feature] == value)[0]
                subset_X = X[subset_indices]
                subset_y = y[subset_indices]
                if len(subset_X) < self.min_samples_leaf:
                    tree[best_feature][value] = np.bincount(y).argmax()
                else:
                    tree[best_feature][value] = self._create_tree(subset_X, subset_y, remaining_features, depth + 1)
            return tree

   # 预测
    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            node = self.tree
            while isinstance(node, dict):
                feature = list(node.keys())[0]
                value = X[i, feature]
                node = node[feature][value]
            predictions.append(node)
        return predictions

    # 计算准确率
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

def main():
    # 导入数据
    data = pd.read_csv("dataset/data_word2.csv", encoding='gbk')
    X = data.drop('label', axis=1).values
    y = data['label'].values

    # 模型
    model = C45(max_depth=5, min_samples_split=5, min_samples_leaf=2)

    # 训练
    model.fit(X, y)

    # 准确率
    acc = model.accuracy(X, y)

    print(f"Acc: {acc}")

if __name__ == '__main__':
    main()
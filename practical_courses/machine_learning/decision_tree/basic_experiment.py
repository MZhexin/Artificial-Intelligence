# 机器学习基础实验课程 实验六
'''
    基础实验内容：
        分别利用ID3和C4.5决策树模型，根据实验数据进行训练；
        计算准确率，对比两种方法的结果。
'''

# 导库
import numpy as np
import pandas as pd

# ID3
class ID3():
    def __init__(self):
        pass

    def fit(self, X, y):
        self.tree = self._create_tree(X, y)

    # 计算熵
    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    #计算信息增益
    def _information_gain(self, X, y, feature_idx):
        entropy_parent = self._entropy(y)
        classes, counts = np.unique(X[:, feature_idx], return_counts=True)
        weighted_entropy = 0
        for c in classes:
            subset_y = y[X[:, feature_idx] == c]
            weighted_entropy += (len(subset_y) / len(y)) * self._entropy(subset_y)

        information_gain = entropy_parent - weighted_entropy
        return information_gain

    #创建决策树（递归创建）
    def _create_tree(self, X, y, depth=0):
        # 判断：若为叶子节点，则返回类别的整型标签；若为子节点，则返回字典
        if len(np.unique(y)) == 1:
            return y[0]
        elif X.shape[1] == 0:
            return np.bincount(y).argmax()
        else:
            best_feature = np.argmax([self._information_gain(X, y, i) for i in range(X.shape[1])])
            tree = {best_feature: {}}
            for value in np.unique(X[:, best_feature]):
                subset_X = X[X[:, best_feature] == value]
                subset_y = y[X[:, best_feature] == value]
                tree[best_feature][value] = self._create_tree(subset_X, subset_y, depth +1)
            return tree

    #预测
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

# C4.5决策树
class C45():
    def __init__(self):
        pass

    def fit(self, X, y):
        self.tree = self._create_tree(X, y)

    # 计算熵
    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)                                                              # 获取y的唯一值和对应的计数
        probabilities = counts / len(y)                                                                                 # 计算每个类别的概率
        entropy = -np.sum(probabilities * np.log2(probabilities))                                                       # 计算熵值
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

    # 创建决策树（递归创建）
    def _create_tree(self, X, y, features=None):
        if features is None:                                                                                            # 如果特征列表为空，则使用所有特征
            features = list(range(X.shape[1]))

        if len(np.unique(y)) == 1:                                                                                      # 如果目标变量只有一个唯一值，则返回该值作为叶节点
            return y[0]
        elif len(features) == 0:                                                                                        # 如果特征列表为空，则返回出现次数最多的类别作为叶节点
            return np.bincount(y).argmax()
        else:
            best_feature = max(features, key=lambda i: self._information_gain_ratio(X, y, i))                           # 选择信息增益比最大的特征作为最佳划分特征
            tree = {best_feature: {}}                                                                                   # 创建以最佳划分特征为根的树
            remaining_features = [f for f in features if f != best_feature]                                             # 获取剩余的特征列表
            for value in np.unique(X[:, best_feature]):                                                                 # 对于最佳划分特征的每一个唯一值
                subset_indices = np.where(X[:, best_feature] == value)[0]                                               # 获取该唯一值对应的子集的索引
                subset_X = X[subset_indices]                                                                            # 获取子集的特征
                subset_y = y[subset_indices]                                                                            # 获取子集的目标变量
                if len(subset_X) == 0:                                                                                  # 如果子集为空，则返回出现次数最多的类别作为叶节点
                    tree[best_feature][value] = np.bincount(y).argmax()
                else:                                                                                                   # 否则，递归地创建子树
                    tree[best_feature][value] = self._create_tree(subset_X, subset_y, remaining_features)
            return tree                                                                                                 # 返回创建的决策树

    # 预测
    def predict(self, X):
        predictions = []                                                                                                # 初始化一个空列表，用于存储预测的结果
        for i in range(X.shape[0]):                                                                                     # 遍历输入特征矩阵的每一行，i是当前的行索引
            node = self.tree                                                                                            # 获取当前决策树的根节点
            while isinstance(node, dict):                                                                               # 当当前节点是一个字典时（即还没有达到叶节点）
                feature = list(node.keys())[0]                                                                          # 获取当前节点的特征名称
                value = X[i, feature]                                                                                   # 获取输入特征矩阵中第i行对应特征的值
                node = node[feature][value]                                                                             # 根据特征名称和特征值，获取该特征划分下的下一个节点
            predictions.append(node)                                                                                    # 将当前节点的值添加到预测结果的列表中
        return predictions                                                                                              # 返回预测结果的列表

    # 计算准确率
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

def main():
    # 导入数据
    data = pd.read_csv("dataset/data_word2.csv", encoding='gbk')
    X = data.drop('label', axis=1).values
    y = data['label'].values

    # 构建ID3决策树和C4.5决策树
    model_ID3 = ID3()
    model_C45 = C45()

    # 分类
    model_ID3.fit(X, y)
    model_C45.fit(X, y)

    # 计算准确率
    acc_ID3 = model_ID3.accuracy(X, y)
    acc_C45 = model_C45.accuracy(X, y)

    # 打印结果
    print(f"ACC of ID3 Decision Tree: {acc_ID3}")
    print(f"ACC of C45 Decision Tree: {acc_C45}")

if __name__ == '__main__':
    main()
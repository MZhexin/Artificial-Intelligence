# 机器学习基础实验课程 实验八
'''
    附加实验内容（1）：
        任选两种核函数重新训练数据，计算相应的分类实验结果
'''

# 导库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 加载数据集
file_path = 'dataset/UCI Heart Disease Dataset.csv'
data = pd.read_csv(file_path)

# 数据预处理：将数据集分为特征（X）和目标变量（y）
X = data.drop('target', axis=1)
y = data['target']

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化核函数名称的列表和存放准确率的字典
kernels = ['linear', 'rbf', 'poly']
accuracies = {}

for kernel in kernels:
    # 初始化并训练SVM模型
    svm_model = SVC(kernel=kernel, random_state=42)
    svm_model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = svm_model.predict(X_test)

    # 计算并保存分类准确率
    accuracies[kernel] = accuracy_score(y_test, y_pred)

# 输出不同核函数的分类准确率
for kernel, accuracy in accuracies.items():
    print(f'{kernel}核函数的分类准确率: {accuracy}')
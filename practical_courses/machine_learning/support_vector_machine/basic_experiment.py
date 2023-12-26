# 机器学习基础实验课程 实验八
'''
    基础实验内容：
        利用线性支持向量机方法对实验数据进行训练；
        实现二分类，计算分类准确率
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

# 初始化并训练线性SVM模型
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = svm_model.predict(X_test)

# 计算分类准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'分类准确率: {accuracy}')
# 机器学习基础实验课程 实验五
'''
    附加实验内容：
        对比高斯贝叶斯、多项式贝叶斯的实验结果
        可自行寻找实验数据，进行扩展实验。
'''

'''
    调库实现
'''

# 导库
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# 读取训练集和测试集数据
train_data = pd.read_csv('dataset/train.csv', header=None)
test_data = pd.read_csv('dataset/test.csv', header=None)

# 设置列名
train_data.columns = ['Label', 'Title', 'Description']
test_data.columns = ['Label', 'Title', 'Description']

train_samples = train_data.sample(n=400, random_state=42)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(train_samples[['Title', 'Description']], train_samples['Label'], test_size=0.2, random_state=42)

# 使用CountVectorizer进行词频统计
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train['Title'] + ' ' + X_train['Description'])
X_val_vec = vectorizer.transform(X_val['Title'] + ' ' + X_val['Description'])

# 使用GaussianNB建立模型
model = GaussianNB()
model.fit(X_train_vec.toarray(), y_train)

# 在测试集上进行预测
X_test_vec = vectorizer.transform(test_data['Title'] + ' ' + test_data['Description'])
y_pred = model.predict(X_test_vec.toarray())

# 计算正确率
accuracy = accuracy_score(test_data['Label'], y_pred)
print(f"Accuracy: {accuracy:.2%}")

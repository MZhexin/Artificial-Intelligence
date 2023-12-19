# 机器学习基础实验课程 实验五
'''
    基础实验内容：
        建立朴素贝叶斯模型，根据实验数据进行训练；
        加入拉普拉斯平滑重新进行训练；
        计算准确率，分析拉普拉斯平滑对结果的影响。

    附加实验内容：
        对比高斯贝叶斯、多项式贝叶斯的实验结果
        可自行寻找实验数据，进行扩展实验。
'''

'''
    备注：在舍友程序的基础上修改的
'''

# 导库
import re
import string
import numpy as np
import pandas as pd
from collections import defaultdict

# 数据预处理函数
def clean_and_tokenize(text):
    text = text.lower()                                                                                                 # 转换为小写
    text = re.sub(f"[{string.punctuation}]", " ", text)                                                                 # 移除标点
    tokens = text.split()                                                                                               # 分词
    return tokens

# 朴素贝叶斯模型训练函数
def train_naive_bayes(train_data, laplace_smoothing=True):
    model = {
        'prior_prob': {},
        'word_prob': defaultdict(lambda: defaultdict(lambda: 0)),
        'vocab': set(),
        'class_counts': defaultdict(lambda: 0)
    }
    total_docs = len(train_data)
    for _, row in train_data.iterrows():
        category = row.iloc[0]
        text = row['text']
        model['class_counts'][category] += 1
        for word in text:
            model['vocab'].add(word)
            model['word_prob'][category][word] += 1
    for category in model['class_counts']:
        model['prior_prob'][category] = model['class_counts'][category] / total_docs
    for category in model['word_prob']:
        total_words = sum(model['word_prob'][category].values())
        vocab_size = len(model['vocab'])
        for word in model['vocab']:
            if laplace_smoothing:
                model['word_prob'][category][word] = (model['word_prob'][category][word] + 1) / (total_words + vocab_size)
            else:
                model['word_prob'][category][word] /= total_words
    return model

# 预测函数
def naive_bayes_predict(text, model):
    class_probs = {category: np.log(prob) for category, prob in model['prior_prob'].items()}
    for category in class_probs:
        for word in text:
            class_probs[category] += np.log(model['word_prob'][category].get(word, 1 / (sum(model['class_counts'].values()) + len(model['vocab']))))
    return max(class_probs, key=class_probs.get)

# 准确率计算函数
def calculate_accuracy(test_data, model):
    correct_predictions = 0
    for _, row in test_data.iterrows():
        category = row.iloc[0]
        text = row['text']
        predicted_category = naive_bayes_predict(text, model)
        if predicted_category == category:
            correct_predictions += 1
    return correct_predictions / len(test_data)

def main():
    # 加载数据
    train_path = 'dataset/train.csv'
    test_path = 'dataset/test.csv'
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # 数据预处理
    train_data['text'] = (train_data.iloc[:, 1] + ' ' + train_data.iloc[:, 2]).apply(clean_and_tokenize)
    test_data['text'] = (test_data.iloc[:, 1] + ' ' + test_data.iloc[:, 2]).apply(clean_and_tokenize)

    # 训练模型
    model_with_smoothing = train_naive_bayes(train_data, laplace_smoothing=True)
    model_without_smoothing = train_naive_bayes(train_data, laplace_smoothing=False)

    # 计算准确率
    accuracy_with_smoothing = calculate_accuracy(test_data, model_with_smoothing)
    accuracy_without_smoothing = calculate_accuracy(test_data, model_without_smoothing)

    # 输出准确率
    print(f"Acc with Laplace Smoothing: {accuracy_with_smoothing}")
    print(f"Acc without Laplace Smoothing: {accuracy_without_smoothing}")


# 主程序
if __name__ == "__main__":
    main()

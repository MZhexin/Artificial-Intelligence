# 机器学习基础实验课程 实验四
'''
    附加实验内容：
        选用不同激活函数进行实验；
        对比不同隐层数量的实验结果；
        可自行寻找实验数据，进行扩展实验。
'''

'''
    注：部分参考ChatGPT
'''

# 导库
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

def main():
    # 加载iris数据集
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # 创建模型和优化器
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 训练模型
    epochs = 100
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = nn.CrossEntropyLoss()(outputs, y_train)
        loss.backward()
        optimizer.step()

    # 在测试集上评估模型
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
        print(f'ACC: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
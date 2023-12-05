# 机器学习基础实验课程 实验四
'''
    基础实验内容：
        建立神经网络模型，根据实验数据进行训练；
        选取不同隐层节点数进行训练；
        计算准确率，分析不同隐层节点数量对结果的影响。
'''

'''
    参考资料：
        （1）https://zhuanlan.zhihu.com/p/115571464
        （2）我舍友的程序
'''

# 导库
import numpy as np
from tqdm import tqdm
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 神经网络
class Net:
    def __init__(self, input_size, hidden_size, output_size):
        self.w_hidden = np.random.rand(input_size, hidden_size)
        self.b_hidden = np.zeros((1, hidden_size))
        self.w_output = np.random.rand(hidden_size, output_size)
        self.b_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, x):
        self.hidden_input = np.dot(x, self.w_hidden) + self.b_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.w_output) + self.b_output
        self.final_output = self.softmax(self.final_input)
        return self.final_output

    def backward(self, x, y, lr):
        samples, features = x.shape

        # 计算输出层的误差和梯度
        one_hot_y = np.zeros((samples, self.final_output.shape[1]))
        one_hot_y[range(samples), y] = 1
        output_error = self.final_output - one_hot_y
        output_grad = output_error / samples

        # 更新输出层的参数
        self.w_output -= lr * np.dot(self.hidden_output.T, output_grad)
        self.b_output -= lr * np.sum(output_grad, axis=0, keepdims=True)

        # 计算隐藏层的误差和梯度
        hidden_error = np.dot(output_grad, self.w_output.T) * self.hidden_output * (1 - self.hidden_output)
        hidden_grad = hidden_error / samples

        # 更新隐藏层的参数
        self.w_hidden -= lr * np.dot(x.T, hidden_grad)
        self.b_hidden -= lr * np.sum(hidden_grad, axis=0, keepdims=True)

    def train(self, x, y, epochs, lr):
        for epoch in tqdm(range(epochs)):
            output = self.forward(x)
            self.backward(x, y, lr)

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

# 主函数
def main():
    # 导入数据
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=22)

    # 数据预处理
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 设置输入层、隐藏层和输出层大小
    input_size = x_train.shape[1]
    hidden_size1 = 10
    hidden_size2 = 100
    hidden_size3 = 1000
    output_size = len(np.unique(y_train))

    # 实例化模型并训练
    # 10个隐藏层节点
    model1 = Net(input_size, hidden_size1, output_size)
    model1.train(x_train, y_train, epochs=2000, lr=0.1)

    # 100个隐藏层节点
    model2 = Net(input_size, hidden_size2, output_size)
    model2.train(x_train, y_train, epochs=2000, lr=0.1)

    # 1000个隐藏层节点
    model3 = Net(input_size, hidden_size3, output_size)
    model3.train(x_train, y_train, epochs=2000, lr=0.1)

    # 预测并评估模型
    y_pred1 = model1.predict(x_test)
    accuracy1 = accuracy_score(y_test, y_pred1)
    print(f"Accuracy (10 Hidden Neurons): {accuracy1}")

    y_pred2 = model2.predict(x_test)
    accuracy2 = accuracy_score(y_test, y_pred2)
    print(f"Accuracy (100 Hidden Neurons): {accuracy2}")

    y_pred3 = model3.predict(x_test)
    accuracy3 = accuracy_score(y_test, y_pred3)
    print(f"Accuracy (1000 Hidden Neurons): {accuracy3}")

if __name__ == '__main__':
    main()
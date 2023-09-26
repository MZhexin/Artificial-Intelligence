# 实验：手写数字识别

# 导入库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 参数初始化
img_size = 8              # 设置图像的长宽
n_mid = 16                # 隐含层神经元个数
n_out = 10                # 输出层神经元个数
learning_rate = 0.01      # 学习率
batch_size = 32           # 批量大小
epoch = 50                # 训练轮数
interval = 5              # 每训练5个批量打印一次结果

digits_data = datasets.load_digits()   # 加载手写数字数据集

# 输入数据
inputs_data = np.asarray(digits_data.data)     # 转换成cupy数组形式以使用GPU加速
inputs_data = (inputs_data - np.average(inputs_data)) / np.std(inputs_data)   # 输入数据初始化（均值为0，标准偏差为1）

# 用独热编码表示正确数据
correct = np.asarray(digits_data.target)
correct_data = np.zeros((len(correct), n_out))
for i in range(len(correct)):
    correct_data[i, correct[i]] = 1

# 用train_test_split()函数划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(inputs_data, correct_data)

# 全连接父层
class Base_Layer(object):
    def update(self, learning_rate):
        self.w -= learning_rate * self.grad_w
        self.b -= learning_rate * self.grad_b

# 中间层
class Middle_Layer(Base_Layer):
    def __init__(self, n_upper, n):    # n_upper表示上一层神经元个数，n表示当前层神经元个数
        self.w = np.random.rand(n_upper, n) * np.sqrt(2 / n_upper)    # He初始化值
        self.b = np.zeros(n)

    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u)   # ReLU函数

    def backward(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1)   # ReLU函数的微分
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta,  axis = 0)
        self.grad_x = np.dot(delta, self.w.T)

# 输出层
class Output_Layer(Base_Layer):
    def __init__(self, n_upper, n):
        self.w = np.random.rand(n_upper, n) * np.sqrt(2 / n_upper)
        self.b = np.zeros(n)

    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.exp(self.u) / np.sum(np.exp(self.u), axis = 1, keepdims = True)  # Sigmoid函数

    def backward(self, target):
        delta = self.y - target
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)

# 初始化网络层
layers = [Middle_Layer(img_size ** 2, n_mid),   # 第一个隐含层
          Middle_Layer(n_mid, n_mid),           # 第二个隐含层
          Output_Layer(n_mid, n_out)]              # 输出层

# 正向传播
def forward_propagation(x):
    for layer in layers:
        layer.forward(x)
        x = layer.y
    return x

# 反向传播
def backward_propagation(target):
    grad_y = target
    for layer in reversed(layers):    # 用reversed()函数使迭代器反向
        layer.backward(grad_y)
        grad_y = layer.grad_x
    return grad_y

# 参数更新
def update_params():
    for layer in layers:
        layer.update(learning_rate)

# 测定误差
def get_error(x, target):
    y = forward_propagation(x)
    return -np.sum(target * np.log(y + 1e-7)) / len(y)    # 交叉熵损失函数

# 测定准确率
'''
   因为标签使用独热编码储存的，只有0和1
   一旦序列中最大值（即1）对应的位置相同，就表明y和target表示的数字是同一个
   再用np.sum()函数计数，即可统计出预测正确值的个数
'''
def get_accuracy(x, target):
    y = forward_propagation(x)
    count = np.sum(np.argmax(y, axis = 1) == np.argmax(target, axis = 1))
    return count / len(y)

# 记录误差
error_record_train = []
error_record_test = []

# 每轮epoch的批次大小
n_batch = len(x_train) // batch_size

for i in range(epoch):
    # 打乱索引
    index_random = np.arange(len(x_train))
    np.random.shuffle(index_random)
    for j in range(n_batch):

        # 提取小批次数据
        md_index = index_random[j * batch_size: (j + 1) * batch_size]
        x_md = x_train[md_index, :]
        y_md = y_train[md_index, :]

        # 正反传播
        forward_propagation(x_md)
        backward_propagation(y_md)

        # 参数更新
        update_params()

    # 误差的测量与记录
    error_train = get_error(x_train, y_train)
    error_record_train.append(error_train)
    error_test = get_error(x_test, y_test)
    error_record_test.append(error_test)

    # 训练过程可视化（打印）
    if i % interval == 0:
        print("Epoch:" + str(i + 1) + "/" + str(epoch),
              "Error_train" + str(error_train),
              "Error_test" + str(error_test))

# 准确率测定
acc_train = get_accuracy(x_train, y_train)
acc_test = get_accuracy(x_test, y_test)
print("Acc_train:" + str(acc_train * 100) + "%",
      "Acc_test:" + str(acc_test * 100) + "%")

# 训练结果可视化（图表）
plt.plot(range(1, len(error_record_train) + 1), error_record_train, label = "Train")
plt.plot(range(1, len(error_record_test) + 1), error_record_test, label = "Test")
plt.legend()

plt.xlabel("Epoches")
plt.ylabel("Error")
plt.show()
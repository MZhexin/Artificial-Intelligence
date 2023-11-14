# 模式识别实验2：概率密度估计
'''
    1．	实现公开的MNIST手写数字数据集的分类或自选数据的分类；包括数据获取、训练集和测试集创建、工作空间创建、训练数据导入；
    2．	调用scikit-learn的基本库，自己实现利用Parzen窗、Gaussian分布和KNN实现MINIST的概率密度估计并进行分类,及基于训练好模型的测试、实验报告撰写；
    3．	利用第三方优化工具包实现参数的优化，并与scikit-learn中标准算法进行自己实现算法的时间复杂度的对比。
'''

# 导库
import tqdm
import time
import numpy as np
from struct import unpack
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# 读取图像
def load_idx3_ubyte(path):
    with open(path, 'rb') as f:                                                                                         # 打开文件
        magic, num, rows, cols = unpack('>4I', f.read(16))                                                              # 幻数、图像数量、单图行列大小
        return np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)                                                  # 读取二进制数据

# 读取标签
def load_idx1_ubyte(path):
    with open(path, 'rb') as f:                                                                                         # 打开文件
        magic, num = unpack('>2I', f.read(8))                                                                           # 幻数、图像数量
        return np.fromfile(f, dtype=np.uint8)                                                                           # 读取二进制数据

# 像素归一化
def normalize_image(image):
    return image.astype(np.float32) / 255.0

# Parzen窗方法：运行时间较长
def parzen(x_train, y_train, x_test, y_test, h=0.1):
    correct = 0
    start_time = time.time()
    for i in tqdm.tqdm(range(len(x_test))):
        Test = x_test[i]
        post_prob = np.zeros(10)

        for j in range(len(x_train)):
            d = np.linalg.norm(Test - x_train[j]) / h
            w = (1 / (np.sqrt(2 * np.pi) * h)) * np.exp(-0.5 * d)
            post_prob[y_train[j]] += w

        Predict = np.argmax(post_prob)
        if Predict == y_test[i]:
            correct += 1
    end_time = time.time()

    accuracy = correct / len(x_test)
    time_cost = end_time - start_time

    return accuracy, time_cost

# 高斯分布方法：调包
def gaussian(x_train, y_train, x_test, y_test):
    start_time = time.time()
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    predictions = gnb.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    end_time = time.time()
    time_cost = end_time - start_time

    return accuracy, time_cost

# K近邻法：调包
def knn(x_train, y_train, x_test, y_test, K=3):
    start_time = time.time()
    knn_classifier = KNeighborsClassifier(n_neighbors=K, weights='distance')
    knn_classifier.fit(x_train, y_train)
    knn_predictions = knn_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, knn_predictions)
    end_time = time.time()
    time_cost = end_time - start_time
    return accuracy, time_cost

def main():
    # 读取数据
    train_images = load_idx3_ubyte('dataset/mnist/train-images.idx3-ubyte')                                             # 读取训练图像
    train_labels = load_idx1_ubyte('dataset/mnist/train-labels.idx1-ubyte')                                             # 读取训练标签
    test_images = load_idx3_ubyte('dataset/mnist/t10k-images.idx3-ubyte')                                               # 读取测试图像
    test_labels = load_idx1_ubyte('dataset/mnist/t10k-labels.idx1-ubyte')                                               # 读取测试标签

    # 归一化
    train_images = normalize_image(train_images)                                                                        # 训练图像归一化
    test_images = normalize_image(test_images)                                                                          # 测试图像归一化

    # 手动实现
    ACC_parzen, Time_parzen = parzen(train_images, train_labels, test_images, test_labels, h=0.1)
    ACC_gaussian, Time_gaussian = gaussian(train_images, train_labels, test_images, test_labels)
    ACC_knn, Time_knn = knn(train_images, train_labels, test_images, test_labels, K=3)

    print('********** Accuracy of Models **********')
    print('Parzen:\t {0}'.format(ACC_parzen))
    print('Gaussian:\t {0}'.format(ACC_gaussian))
    print('KNN:\t {0}'.format(ACC_knn))
    print('********** Time Cost of Models **********')
    print('Parzen:\t {0}'.format(Time_parzen))
    print('Gaussian:\t {0}'.format(Time_gaussian))
    print('KNN:\t {0}'.format(Time_knn))

if __name__ == '__main__':
    main()
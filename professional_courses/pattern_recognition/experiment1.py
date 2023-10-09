# 实验1：基于Bayes的手写字符识别
# 代码参考：http://t.csdnimg.cn/vU2Ev

# 导库
import numpy as np
from struct import unpack                                                                                   # 用来处理二进制数据

# 读取图像
def load_idx3_ubyte(path):
    with open(path, 'rb') as f:                                                                             # 打开文件
        magic, num, rows, cols = unpack('>4I', f.read(16))                                                  # 幻数、图像数量、单图行列大小
        return np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)                                      # 读取二进制数据

# 读取标签
def load_idx1_ubyte(path):
    with open(path, 'rb') as f:                                                                             # 打开文件
        magic, num = unpack('>2I', f.read(8))                                                               # 幻数、图像数量
        return np.fromfile(f, dtype=np.uint8)                                                               # 读取二进制数据

# 像素归一化
def normalize_image(image):
    return image.astype(np.float32) / 255.0

# 特征提取
# 参数：图像、提取完特征后想要的尺寸、阈值
def feature_extraction(images, side_length=28, threshold=0.2):
    num = 28 // side_length                                                                                 # 计算平滑时应该使用的步长（原图像尺寸整除目标尺寸并向下取整）
    features = np.empty((images.shape[0], side_length, side_length), dtype=np.float32)                      # 创建空数组来存储特征
    for i in range(images.shape[0]):                                                                        # 对每张图像
        for j in range(side_length):                                                                        # 对每一行
            for k in range(side_length):                                                                    # 对每一列
                temp_mean = images[i, num * j:num * (j + 1), num * k:num * (k + 1)].mean()                  # 切片取将图像划分成目标尺寸个格子，每个格子的大小为步长×步长的矩阵，使用小矩阵的均值作为该格子的特征
                features[i, j, k] = 1 if temp_mean > threshold else 0                                       # 如果特征大于阈值，赋1（保留）；否则特征值为0（舍弃）
    return features                                                                                         # 返回特征

# 训练朴素贝叶斯模型
'''
    1. 先验概率：图像为某一类的概率
    2. 条件概率：当图像为某一类时，某一处像素值为某一值的概率
'''
def train_naive_bayes(train_features, train_labels, num_classes=10):
    num_samples, side_length, side_length = train_features.shape                                            # 获取样本数量、图像尺寸
    num_pixels = side_length * side_length                                                                  # 每张图像所需的像素个数
    priori_probs = np.zeros(num_classes)                                                                    # 初始化先验概率
    condition = np.zeros((num_classes, num_pixels, 2))                                                      # 初始化条件概率

    # 计算概率时采用拉普拉斯平滑：为防止计算出来某一项的概率为零，让分子加1，分母加上类别个数或某个常数（我们这里取10的-5次方，即1e-5）
    for label in range(num_classes):                                                                        # 对于每个类别
        priori_probs[label] = (np.sum(train_labels == label) + 1) / (num_samples + num_classes)             # 标签为label的先验概率：标签为label的图像在所有图像中的占比
        class_samples = train_features[train_labels == label]                                               # 找到所有标签为label的图像
        for pixel in range(num_pixels):                                                                     # 对于每个像素
            pixel_values = class_samples[:, pixel // side_length, pixel % side_length]                      # 像素值
            condition[label, pixel, 0] = (np.sum(pixel_values == 0) + 1) / (class_samples.shape[0] + 1e-5)  # 像素值为1的条件概率
            condition[label, pixel, 1] = (np.sum(pixel_values == 1) + 1) / (class_samples.shape[0] + 1e-5)  # 像素值为0的条件概率
    return priori_probs, condition

# 对测试图像进行预测
def naive_bayes_predict(test_features, priori_probs, condition):
    num_samples, side_length, side_length = test_features.shape                                             # 获取样本数量、图像尺寸
    num_pixels = side_length * side_length                                                                  # 每张图像所需的像素个数
    num_classes = len(priori_probs)                                                                         # 类别个数
    predictions = np.zeros(num_samples, dtype=int)                                                          # 初始化概率（即预测出每张图像的类别）

    for i in range(num_samples):                                                                            # 对于每张图像
        log_probs = np.zeros(num_classes)                                                                   # 初始化似然概率
        for label in range(num_classes):                                                                    # 对于每个类别
            log_probs[label] = np.log(priori_probs[label])                                                  # 似然概率：对先验概率取以e为底的对数
            for pixel in range(num_pixels):                                                                 # 对于每个像素
                pixel_value = int(test_features[i, pixel // side_length, pixel % side_length])              # 像素值
                log_probs[label] += np.log(condition[label, pixel, pixel_value])                            # 计算似然概率并累加，认为这就是该图像属于某一类的概率

        predictions[i] = np.argmax(log_probs)                                                               # 认为该图像的类别为计算出概率最大的那一类

    return predictions

# 模型分析
def evaluate_model(test_features, test_labels, priori_probs, condition):
    predictions = naive_bayes_predict(test_features, priori_probs, condition)                               # 预测结果
    accuracy = np.mean(predictions == test_labels)                                                          # 准确率
    return accuracy

# 主函数
def main():
    # 读取数据
    train_images = load_idx3_ubyte('dataset/mnist/train-images.idx3-ubyte')                                 # 读取训练图像
    train_labels = load_idx1_ubyte('dataset/mnist/train-labels.idx1-ubyte')                                 # 读取训练标签
    test_images = load_idx3_ubyte('dataset/mnist/t10k-images.idx3-ubyte')                                   # 读取测试图像
    test_labels = load_idx1_ubyte('dataset/mnist/t10k-labels.idx1-ubyte')                                   # 读取测试标签

    # 归一化
    train_images = normalize_image(train_images)                                                            # 训练图像归一化
    test_images = normalize_image(test_images)                                                              # 测试图像归一化

    # 特征提取
    train_features = feature_extraction(train_images)                                                       # 训练图像特征提取
    test_features = feature_extraction(test_images)                                                         # 训练图像特征提取

    priori_probs, condition = train_naive_bayes(train_features, train_labels)                               # 概率
    accuracy = evaluate_model(test_features, test_labels, priori_probs, condition)                          # 准确率

    # 打印准确率
    print(f"ACC: {accuracy * 100:.2f}%")                                                                    # 打印

# 运行
if __name__ == '__main__':
    main()
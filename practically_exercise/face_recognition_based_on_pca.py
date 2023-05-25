# 实验：基于主成分分析（PCA）的人脸识别

# 导库
import os
import cv2
import math
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import scipy.io as scio
from numpy import linalg
from random import shuffle
from collections import Counter
import matplotlib.pyplot as plt

# 定义PCA类
class PCA():
    def __init__(self, data, dimension):
        self.data = data                                                            # 数据
        self.dimension = math.ceil(0.9 * dimension) - 1                             # 计算d的最佳取值（最佳值 ≥ 训练集维度 × 90%）

    def analysis(self):
        data_mean = np.mean(self.data, axis=0)                                      # 计算均值
        data_norm = np.array([obs - data_mean for obs in self.data])                # 标准化（原始数据与均值相减）
        data_cov = np.cov(data_norm, rowvar=False)                                  # 计算协方差矩阵
        eigenvalue, featurevector = np.linalg.eig(np.mat(data_cov))                 # 计算特征值和特征向量
        sorted_index = np.argsort(eigenvalue)                                       # 按特征值大小排序
        eigen_matrix = featurevector[:, sorted_index[-1:-self.dimension:-1]]        # 取前d个最大的特征值，即特征脸空间
        data_sample = np.dot(data_norm, eigen_matrix)                               # 计算样本投影
        return data_mean, eigen_matrix, data_sample                                 # 依次返回均值、特征脸空间和样本投影

# 训练
def train():
        # 初始化参数
        accuracy, correct = 0, 0                                                    # 准确率和正确分类计数器
        train_img, train_label = [], []                                             # 训练集数据和标签
        test_img, test_label = [], []                                               # 测试集数据和标签

        # 数据读取
        img_path = r'data\img.xlsx'                                                 # 图像路径
        label_path = r'data\label.xlsx'                                             # 标签路径
        imgs = pd.read_excel(img_path, header=None).values                          # 读取图像
        labels = pd.read_excel(label_path, header=None).values                      # 读取标签

        # 划分训练集和测试集（按7:4划分）
        for i in range(0, 8):
            for items in imgs[i :: 11]:                                             # 训练集图像
                train_img.append(items)
            for items in labels[i :: 11]:                                           # 训练集标签
                train_label.append(items)
        for i in range(7, 12):
            for items in imgs[i :: 11]:                                             # 测试集图像
                test_img.append(items)
            for items in labels[i :: 11]:                                           # 测试集标签
                test_label.append(items)

        # 数据预处理：注意测试集使用的均值等也来自训练集
        img_train_pca = PCA(train_img, 105)                                         # 训练集图像的PCA对象
        train_mean = img_train_pca.analysis()[0]                                    # 返回均值
        train_space = img_train_pca.analysis()[1]                                   # 返回特征脸空间
        train_sample = img_train_pca.analysis()[2]                                  # 返回样本投影
        test_norm = test_img - train_mean                                           # 标准化测试集
        test_sample = np.dot(test_norm, train_space)                                # 计算测试集样本

        # 计算欧氏距离：利用二范数
        for test_num in range(0, len(test_label)):                                  # 对每个测试集样本，都计算所有特征脸（train）到它的距离
            distance = []                                                           # 对每个测试集样本，都重置一次距离列表
            for train_num in range(0, len(train_label)):
                distance.append(np.linalg.norm(train_sample[train_num] - test_sample[test_num]))

            index = np.argmin(distance)                                             # 返回令distance达到最小值的索引
            predict_value = train_label[index][0]                                   # 预测值（用训练集预测的，因此在train_label中获取索引）
            target_value = test_label[test_num]                                     # 真实值

            if predict_value == target_value:                                       # 若预测正确，计数器加一
                correct += 1

        accuracy = correct / len(test_label)                                       # 计算准确率
        print('ACC: {0}'.format(accuracy))                                          # 打印结果

if __name__ == '__main__':
    train()
    print('结束')

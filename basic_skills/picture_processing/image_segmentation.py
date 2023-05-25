# 实验：图像分割

# 导库
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 迭代阈值分割法
def IterativeThresholdingMethod(img):
    # 初始化
    Image = np.array(img).astype(np.float32)                                # 将图像转化成数组
    zmax = np.max(Image)                                                    # 获取最大值
    zmin = np.min(Image)                                                    # 获取最小值
    initial_threshold = (zmax + zmin) / 2                                   # 设置初始阈值
    threshold = initial_threshold                                           # 每一轮迭代的阈值（先初始化为初值）
    object, background = 0, 0
    object_num, background_num = 0, 0

    while abs(initial_threshold - threshold) > 0:                           # 退出迭代的条件为找到最佳阈值
        for length in range(0, len(Image) - 1):
            for width in range(0, len(Image[0]) - 1):
                if Image[length, width] < threshold:                        # 像素值小于阈值的视为背景
                    background += Image[length, width]
                    background_num += 1
                elif Image[length, width] >= threshold:                     # 像素值大于阈值的视为物体
                    object += Image[length, width]
                    object_num += 1

        # 计算均值
        object_average = object / object_num
        background_average = background / background_num

        object, object_num, background, background_num = 0, 0, 0, 0         # 重置参数
        initial_threshold = threshold                                       # 重置初始阈值为本轮阈值
        threshold = int((object_average + background_average) / 2)          # 计算新阈值

    new_img = img.copy()                                                    # 复制原图像

    for length in range(0, len(Image) - 1):
        for width in range(0, len(Image[0]) - 1):
            if new_img[length, width] < threshold:                          # 像素值小于最佳阈值的视为背景，像素值置0
                new_img[length, width] = 0
            elif Image[length, width] >= threshold:                         # 像素值大于最佳阈值的视为物体，像素值置1
                new_img[length, width] = 255

    # 返回最佳阈值和处理后的图像
    return threshold, new_img

# 读取原图
img = cv2.imread("images/input_images/my_wife.jpg", flags=0)
cv2.imshow('Oriinal Image', img)
key1 = cv2.waitKey()

# # 分割之后
new_img = IterativeThresholdingMethod(img)[1]
cv2.imshow('New Image', new_img)
key2 = cv2.waitKey()

# P.S. 我的发现：图像用opencv-python读出来就是正常的，用matplotlib读出来颜色就变了，原因暂时不详 ————————> 代码如下，感兴趣可以尝试运行
'''
    # 读取原图
    img = cv2.imread("images/input_images/my_wife.jpg", flags=0)
    plt.figure()
    plt.imshow(img)
    plt.title('Original Image')
    plt.show()
    
    # 分割之后
    new_img = IterativeThresholdingMethod(img)[1]
    plt.figure()
    plt.imshow(Image.fromarray(img))
    plt.title('Segmented Image')
    plt.show()
'''

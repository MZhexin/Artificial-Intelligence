# Roberts算子 ————> 代码全部来自CSDN（链接：http://t.csdn.cn/RuaM0）

import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取图像
img = cv2.imread('images/input_images/my_wife.jpg', 0)

# roberts算子
kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
kernely = np.array([[0, -1], [1, 0]], dtype=int)
x = cv2.filter2D(img, cv2.CV_16S, kernelx)
y = cv2.filter2D(img, cv2.CV_16S, kernely)

# 转uint8
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

# 加权和
Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# 显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 图像显示
titles = [u'原始图像', u'Robertes图像']
images = [img, Roberts]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.xticks([]), plt.yticks([])
    plt.title(titles[i])
plt.show()

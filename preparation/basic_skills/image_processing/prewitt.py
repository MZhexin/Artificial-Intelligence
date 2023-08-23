# Prewitt算子 ————> 代码全部来自CSDN（链接：http://t.csdn.cn/RuaM0）

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 输入图像
img = cv2.imread('images/input_images/my_wife.jpg', 0)

# kernel
kernelX = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
kernelY = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
x = cv2.filter2D(img, cv2.CV_16S, kernelX)
y = cv2.filter2D(img, cv2.CV_16S, kernelY)

# 转uint8
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

# 加权和
Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# 图象显示
plt.rcParams['font.sans-serif'] = ['SimHei']
titles = [u'原始图像', u'Prewitt图像']
images = [img, Prewitt]

for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.xticks([]), plt.yticks([])
    plt.title(titles[i])
plt.show()

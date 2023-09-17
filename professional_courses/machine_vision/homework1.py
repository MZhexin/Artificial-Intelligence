# 作业1：图像处理复习
'''
    要求：
    （1）计算图形直方图
    （2）进行直方图均衡化处理
    （3）实现一种图像滤波并输出处理后的图像及直方图
'''

# 导库
import cv2
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('images/elevator.jpg', 0)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 设置画布
plt.figure(figsize=(14, 18))
plt.subplots_adjust(hspace=0.5)

# 原图像及其直方图
plt.subplot(3, 2, 1)
plt.imshow(img)
plt.axis('off')
plt.title('Original Image')
plt.subplot(3, 2, 2)
plt.hist(img.ravel(), 256)
plt.title('Gray Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Proportion')

# 直方图均衡化
equalization = cv2.equalizeHist(image)
equ = cv2.cvtColor(equalization, cv2.COLOR_BGR2RGB)
plt.subplot(3, 2, 3)
plt.imshow(equ)
plt.axis('off')
plt.title('Equalized Image')
plt.subplot(3, 2, 4)
plt.hist(equ.ravel(), 256)
plt.title('Equalized Gray Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Proportion')

# 图像滤波
filtered = cv2.GaussianBlur(img, (5, 5), 0)
plt.subplot(3, 2, 5)
plt.imshow(filtered)
plt.axis('off')
plt.title('Filtered Image')
plt.subplot(3, 2, 6)
plt.hist(filtered.ravel(), 256)
plt.title('Filtered Gray Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Proportion')

# 显示图像
plt.show()




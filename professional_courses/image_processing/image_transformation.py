# 实验：图像变换

# 导库
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('images/my_wife.jpg', 0)  # 读取灰度图像

'''
    知识点：
    1、numpy.fft.fft() 计算一维傅立叶变换
    2、numpy.fft.fft2() 计算二维傅立叶变换
    3、numpy.fft.shift() 将图像中的低频部分移动到图像的中心
'''
img_ft = np.fft.fft2(img)  # 傅立叶变换————>变换完的结果分布在四个角上，需要移频，使频谱图保持在图像正中央
img_ft_shift = np.fft.fftshift(img_ft)  # 移频，使傅立叶变换结果中心化

# 逆变换
img_ift_shift = np.fft.ifftshift(img_ft_shift)
img_ift = np.fft.ifft2(img_ift_shift)
img_i = np.abs(img_ift)

'''
    取绝对值：求img_ft的频谱
    取对数：缩小频谱的范围，同时增大高频部分的亮度
    乘以20：放大一定倍数，使图像更方便分析
'''
s1 = 20 * np.log(np.abs(img_ft))         # 频谱显示
s2 = 20 * np.log(np.abs(img_ft_shift))   # 中心化的频谱显示

cv2.imshow('Original Image', img)        # 显示原图
key0 = cv2.waitKey()

# 利用matplotlib.pyplot库绘制OpenCV的频谱图
plt.imshow(s1, cmap='gray')              # 显示图像，camp='gray'设置灰度参数
plt.title('Magnitude Spectrum')          # 图像标题
plt.xticks([]), plt.yticks([])           # 指定X轴、Y轴刻度，但因为传入的参数是空列表，所以不显示坐标轴
plt.show()                               # 显示图像
key1 = cv2.waitKey()                     # 延长图像出现的时间

# 中心化频谱图同理
plt.imshow(s2, cmap='gray')              # 显示图像，camp='gray'设置灰度参数
plt.title('Centered Magnitude Spectrum') # 图像标题
plt.xticks([]), plt.yticks([])           # 指定X轴、Y轴刻度，但因为传入的参数是空列表，所以不显示坐标轴
plt.show()                               # 显示图像
key2 = cv2.waitKey()                     # 延长图像出现的时间

# 打印逆变换图像
plt.imshow(img_i, cmap='gray')
plt.title('Inverse Transformation Image')
plt.xticks([]), plt.yticks([])
plt.show()
key3 = cv2.waitKey()

# 相位谱
'''
    小知识：numpy.angle()函数用于计算复数对应的角度
'''
img_ph = np.angle(img_ft)                  # 傅里叶变换结果的相位
img_ph_shift = np.angle(img_ft_shift)       # 计算中心化傅立叶变换结果的相位

# 打印相位谱图
plt.imshow(img_ph, cmap='gray')
plt.title('Phase Spectrum')
plt.xticks([]), plt.yticks([])
plt.show()
key4 = cv2.waitKey()

plt.imshow(img_ph_shift, cmap='gray')
plt.title('Centered Phase Spectrum')
plt.xticks([]), plt.yticks([])
plt.show()
key5 = cv2.waitKey()

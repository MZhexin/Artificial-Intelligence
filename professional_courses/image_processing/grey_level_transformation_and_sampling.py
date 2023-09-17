# 实验：图像灰度变换与空间采样

import cv2
import numpy as np
import torchvision.transforms as transforms

# 读取图像
img_gray = cv2.imread('images/my_wife.jpg', flags=0)  # 灰度图
img_color = cv2.imread('images/my_wife.jpg', flags=1)  # 彩色图

# 显示图像
cv2.imshow('img_gray', img_gray)
cv2.imshow('img_color', img_color)
key1 = cv2.waitKey()   # 延迟图片出现时间，防止因为程序运行过快导致图像只出现一瞬间

# 下采样
dst1 = cv2.resize(img_color, (0,0), None, 0.5, 0.5)  # 1/2
dst2 = cv2.resize(img_color, (0,0), None, 0.25, 0.25)  # 1/4
dst3 = cv2.resize(img_color, (0,0), None, 0.125, 0.125)   # 1/8
dst4 = cv2.resize(img_color, (0,0), None, 0.0625, 0.0625)   # 1/16
dst5 = cv2.resize(img_color, (0,0), None, 0.03125, 0.03125)   # 1/32
# 显示图像
cv2.imshow('原图像', img_color)
cv2.imshow('1/2', dst1)
cv2.imshow('1/4', dst2)
cv2.imshow('1/8', dst3)
cv2.imshow('1/16', dst4)
cv2.imshow('1/32', dst5)
key2 = cv2.waitKey()

# 获取图像宽高
size = dst5.shape
height = size[0]  # 高
width = size[1]  # 宽
channel = size[2]  # 通道数
print("图像的宽、高、通道数分别是:{0},{1},{2}".format(width, height, channel))

# 输出区域内的像素值
dst5_tensor = transforms.ToTensor()(dst5).permute(2, 1, 0)
print(dst5_tensor.shape)
for i in range(0, width - 1):
    for j in range(0, height - 1):
        print((dst5_tensor[i, j].numpy() * 255).astype(np.uint8), end='')   # float32转int8再输出
    print('\n')

# 灰度量化
img_color_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)   # 把彩色的老婆变成黑白的老婆
cv2.imshow('img_color_gray.jpg', img_color_gray)
key3 = cv2.waitKey()
dst_4 = np.uint8(img_color_gray / 64) * 64   # 4
dst_8 = np.uint8(img_color_gray / 32) * 32   # 8
dst_16 = np.uint8(img_color_gray / 16) * 16   # 16
dst_256 = np.uint8(img_color_gray / 256) * 256 # 256
cv2.imshow('dat_4', dst_4)
cv2.imshow('dst_8', dst_8)
cv2.imshow('dst_16', dst_16)
cv2.imshow('dst256', dst_256)
key4 = cv2.waitKey()

# 代数运算
my_wife = np.expand_dims(img_gray ,axis=2).repeat(3, axis=2)
img_add = my_wife + img_color
img_min = my_wife - img_color
img_mul = my_wife * img_color
img_dev_temp = np.random.randint(1, 10,(1449, 1080, 3))  # 生成用于除法的模板(取值范围为1-10，防止除数为零）
img_dev = img_color / img_dev_temp
cv2.imshow('add', img_add)
cv2.imshow('min', img_min)
cv2.imshow('mul', img_mul)
cv2.imshow('dev', img_dev)
key5 = cv2.waitKey()

# 逻辑运算
img_and = cv2.bitwise_and(my_wife, img_color)
img_or = cv2.bitwise_or(my_wife, img_color)
img_xor = cv2.bitwise_xor(my_wife, img_color)
img_not = cv2.bitwise_not(img_color)
cv2.imshow('img_and', img_and)
cv2.imshow('img_or', img_or)
cv2.imshow('img_xor', img_xor)
cv2.imshow('img_not', img_not)
key6 = cv2.waitKey()

# BGR与HSV的转换
img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
cv2.imshow('img_hsv', img_hsv)
key7 = cv2.waitKey()

# 实验：形态学图像处理：腐蚀、膨胀与开闭运算

# 导库
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像（文本图像和指纹图像来自学校老师给的数据）
img = cv2.imread("images/input_images/my_wife.jpg", 0)                      # 明日方舟干员初雪的美图
script = cv2.imread('images/input_images/script.tif', 0)                    # 包含字符残缺的文本图像
fingerprint = cv2.imread('images/input_images/fingerprint.tif', 0)          # 含有杂点的指纹图像

# 设置卷积核
kernel = np.ones((3, 3),np.uint8)

img_eroded = cv2.erode(img, kernel, iterations=1)                           # 腐蚀，参数iterations表示迭代次数，即腐蚀多少次
script_dilate = cv2.dilate(script, kernel, iterations=1)                    # 膨胀，参数iterations作用同上

# 开闭运算处理含有杂点的指纹图像
fingerprint_open = cv2.morphologyEx(fingerprint, cv2.MORPH_OPEN, kernel)    # 开运算：先腐蚀再膨胀
fingerprint_close = cv2.morphologyEx(fingerprint, cv2.MORPH_CLOSE, kernel)  # 闭运算：先膨胀再腐蚀

# 显示各个图像
cv2.imshow('Original Image', img)
cv2.imshow('Original Script', script)
cv2.imshow('Original Fingerprint', fingerprint)
cv2.imshow('Eroded Image', img_eroded)
cv2.imshow('Dilated Script', script_dilate)
cv2.imshow('Open', fingerprint_open)
cv2.imshow('Close', fingerprint_close)
key = cv2.waitKey()

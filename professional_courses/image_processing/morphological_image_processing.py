# 实验：形态学图像处理：腐蚀、膨胀与开闭运算

# 导库
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像（文本图像和指纹图像来自学校老师给的数据）
img = cv2.imread("images/my_wife.jpg", 0)                      # 明日方舟干员初雪的美图
script = cv2.imread('images/script.tif', 0)                    # 包含字符残缺的文本图像
fingerprint = cv2.imread('images/fingerprint.tif', 0)          # 含有杂点的指纹图像

# 设置卷积核
kernel = np.ones((3, 3),np.uint8)

img_eroded = cv2.erode(img, kernel, iterations=1)                           # 腐蚀，参数iterations表示迭代次数，即腐蚀多少次
script_dilate = cv2.dilate(script, kernel, iterations=1)                    # 膨胀，参数iterations作用同上

# 开闭运算处理含有杂点的指纹图像（调用opencv-python库）
fingerprint_open = cv2.morphologyEx(fingerprint, cv2.MORPH_OPEN, kernel)    # 开运算：先腐蚀再膨胀
fingerprint_close = cv2.morphologyEx(fingerprint, cv2.MORPH_CLOSE, kernel)  # 闭运算：先膨胀再腐蚀

# 开闭运算处理含有杂点的指纹图像（利用开闭运算的定义与腐蚀、膨胀运算的代码）
fingerprint_open_ed = cv2.dilate(cv2.erode(fingerprint, kernel, iterations=1), kernel, iterations=1)
fingerprint_close_ed = cv2.erode(cv2.dilate(fingerprint, kernel, iterations=1), kernel, iterations=1)

# 显示各个图像
cv2.imshow('Original Image', img)
cv2.imshow('Original Script', script)
cv2.imshow('Original Fingerprint', fingerprint)
cv2.imshow('Eroded Image', img_eroded)
cv2.imshow('Dilated Script', script_dilate)
cv2.imshow('Open1', fingerprint_open)
cv2.imshow('Open2', fingerprint_open_ed)
cv2.imshow('Close1', fingerprint_close)
cv2.imshow('Close2', fingerprint_close_ed)
key = cv2.waitKey()

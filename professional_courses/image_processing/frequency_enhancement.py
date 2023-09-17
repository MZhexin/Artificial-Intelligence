# 实验： 图像频域增强

# 导库
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# n阶巴特沃斯低通滤波器
def blpf(image, d, n):

    f = np.fft.fft2(image)  # 对图像做傅里叶变换
    fshift = np.fft.fftshift(f)  # 移频————>目的：中心化
    transfor_matrix = np.zeros(image.shape)  # 设置变换矩阵（全零矩阵）
    M = transfor_matrix.shape[0]  # 第一维
    N = transfor_matrix.shape[1]  # 第二维

    # 设置n阶巴特沃斯低通滤波器 ——————> H_BLPF(u, v) = 1 / 1 + [ D(u, v) / D_0 ] ^ 2n
    for u in range(M):
        for v in range(N):
            D = math.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)  # D(u, v)
            transfor_matrix[u, v] = 1 / (1 + pow(D / d, 2 * n))  # 最终的滤波器

    # 新图像
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * transfor_matrix)))  # 取绝对值
    return new_img  # 返回新图像

img = cv2.imread('images/my_wife.jpg')  # 读取图像
# N阶巴特沃斯低通滤波器
img_blpf_1 = blpf(img, 20, 1)  # 一阶
img_blpf_2 = blpf(img, 20, 2)  # 二阶
img_blpf_5 = blpf(img, 20, 5)  # 五阶

# 打印图像
plt.imshow(img)
plt.title('original')
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(img_blpf_1)
plt.title('img_blpf_1')
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(img_blpf_2)
plt.title('img_blpf_2')
plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(img_blpf_5)
plt.title('img_blpf_5')
plt.xticks([]), plt.yticks([])
plt.show()

key1 = cv2.waitKey()
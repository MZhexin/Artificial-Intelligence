# 实验：图像分割

# 导库
import cv2
import math
import numpy as np

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

# Otsu阈值分隔法
def OtsuMethod(img):
    # 初始化1
    rows, cols = img.shape[:2]                                              # 获取行、列
    gray_hist = cv2.calcHist([img], [0], None, [256], [0, 256])             # 计算灰度直方图（直接引用opencv库函数，注意参数需要加中括号，否则会报错）
    norm_hist = gray_hist / float(rows * cols)                              # 归一化灰度直方图
    zero_cumu_moment = np.zeros([256], np.float32)                          # 零阶累积距，即累加直方图，P(rk) = n1 + n2 + … + nk
    one_cumu_moment = np.zeros([256], np.float32)                           # 一阶累积距，即图像总体的灰度平均值

    # 计算零阶累积矩, 一阶累积矩
    '''
        第i元素的累积距需要加上i-1个元素的累积距，但又因每一个元素都存在数组对应的下标中，所以不能使用+=符号写
        初始化的每一个zero_cumu_moment[i]都是0，用+=后就变成了zero_cumu_moment[i] = 0 + norm_hist[i]，与事实不符
    '''
    for i in range(256):
        if i == 0:
            zero_cumu_moment[i] = norm_hist[i]
            one_cumu_moment[i] = 0
        else:
            zero_cumu_moment[i] = zero_cumu_moment[i - 1] + norm_hist[i]
            one_cumu_moment[i] = one_cumu_moment[i - 1] + i * norm_hist[i]

    # 初始化2
    mean = one_cumu_moment[255]                                             # 均值
    variance = np.zeros([256], np.float32)                                  # 方差

    # 计算每一个灰度级作为阈值时的方差
    for i in range(0, 256):
        variance[i] = math.pow(mean * zero_cumu_moment[i] - one_cumu_moment[i], 2) / (zero_cumu_moment[i] * (1.0 - zero_cumu_moment[i]))

    # 找到最大的方差对应的阈值
    threshold = np.argmax(variance)

    new_img = img.copy()                                                    # 复制原图像
    Image = np.array(img).astype(np.float32)                                # 将图像转化成数组

    # 阈值分割
    for length in range(0, len(Image) - 1):
        for width in range(0, len(Image[0]) - 1):
            if new_img[length, width] < threshold:
                new_img[length, width] = 0
            elif Image[length, width] >= threshold:
                new_img[length, width] = 255

    # 返回最佳阈值和处理后的图像
    return threshold, new_img

# 读取原图
img = cv2.imread("images/input_images/my_wife.jpg", flags=0)
cv2.imshow('Oriinal Image', img)
key1 = cv2.waitKey()

# 迭代阈值分割法
new_img1 = IterativeThresholdingMethod(img)[1]
cv2.imshow('Interative Threshold Method', new_img1)
key2 = cv2.waitKey()

# Otsu阈值分割法
new_img2 = OtsuMethod(img)[1]                                               # 用自己写的Otsu方法
new_img3 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)[1]                   # 用opencv库自带的Otsu方法
cv2.imshow('Otsu Method', new_img2)
cv2.imshow('Otsu Method(Use Function)', new_img3)
key3 = cv2.waitKey()

# P.S. 我的发现：图像用opencv-python读出来就是正常的，用matplotlib读出来颜色就变了，原因暂时不详 ————————> 代码如下，感兴趣可以尝试运行
'''
    # 读取原图
    img = cv2.imread("images/input_images/my_wife.jpg", flags=0)
    plt.figure()
    plt.imshow(img)
    plt.title('Original Image')
    plt.show()
    
    # 分割之后
    new_img1 = IterativeThresholdingMethod(img)[1]
    plt.figure()
    plt.imshow(Image.fromarray(new_img1))
    plt.title('Segmented Image')
    plt.show()
'''


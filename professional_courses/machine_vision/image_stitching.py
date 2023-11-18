# 机器视觉作业2：拍摄校园两幅有重叠部分的照片，编写代码完成两幅图像拼接
'''
    要求：
        1）选择一种特征点检测算法，在两幅图像上检测特征点
        2）选择图像点特征的表示方法，通过特征比对实现两幅图像对应特征点的获取
        3）选择合适的变换，通过对应特征点计算变换参数，实现图像的拼接
'''

# 导库
import cv2
import numpy as np

# 特征点检测
def detection(image):
    sift = cv2.SIFT_create()                                                                                            # 创建SIFT算子
    keypoints, descriptors = sift.detectAndCompute(image, None)                                                         # 返回特征点和描述子
    return keypoints, descriptors

# 特征点匹配
def matching(des1, des2):
    matcher = cv2.BFMatcher()                                                                                           # 匹配算法：暴力匹配法
    matches = matcher.knnMatch(des1, des2, k=2)                                                                         # 返回最近的两个特征点

    # 小于一定距离的特征点认为匹配成功
    good_matches = []                                                                                                   # 初始化匹配点列表
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches


# 图像拼接
def stitching(image1, image2, matches, keypoints1, keypoints2, max_distance=10):
    src_pts = []
    dst_pts = []

    for match in matches:
        src_pts.append(keypoints1[match.queryIdx].pt)                                                                   # 主动匹配的描述符组中描述符
        dst_pts.append(keypoints2[match.trainIdx].pt)                                                                   # 被匹配的描述符组中描述符

    # 数据处理
    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, max_distance)                                            # 变换矩阵和掩膜

    result = cv2.warpPerspective(image1, M, (image2.shape[1] + image1.shape[1], image2.shape[0]))                       # 透视变换：转变视角
    result[0:image2.shape[0], 0:image2.shape[1]] = image2                                                               # 合并图像
    return result

def main():
    # 读入图像（三种）
    # # 1、毛虫望远镜.jpg
    # image1 = cv2.imread(r'images\homework2\telescope1.jpg')
    # image2 = cv2.imread(r'images\homework2\telescope2.jpg')

    # 2、一輪強勁的音樂響起，好像是一首很舊的歌。
    image1 = cv2.imread(r'images\homework2\the_four1.jpg')
    image2 = cv2.imread(r'images\homework2\the_four2.jpg')

    # 3、校园景象
    # image1 = cv2.imread(r'images\homework2\campus2.jpg')
    # image2 = cv2.imread(r'images\homework2\campus1.jpg')

    # 特征检测
    keypoints1, descriptors1 = detection(image1)
    keypoints2, descriptors2 = detection(image2)

    # 特征匹配
    matches = matching(descriptors1, descriptors2)

    # 图像拼接
    result = stitching(image1, image2, matches, keypoints1, keypoints2)

    # 显示图像
    cv2.imshow('Left Part', image1)
    cv2.imshow('Right Part', image2)
    cv2.imshow('Result', result)
    cv2.waitKey()

if __name__ == '__main__':
    main()
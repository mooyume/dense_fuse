# coding: utf-8
import cv2
import numpy as np
from skimage.measure import shannon_entropy
import os

import matplotlib.pyplot as plt
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 熵
def EN(img):
    return shannon_entropy(img)


# 标准差
def SD(img):
    return np.std(img)


def cross_covariance(x, y, mu_x, mu_y):
    return 1 / (x.size - 1) * np.sum((x - mu_x) * (y - mu_y))


def SSIM(x, y):
    L = np.max(np.array([x, y])) - np.min(np.array([x, y]))
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sig_x = np.std(x)
    sig_y = np.std(y)
    sig_xy = cross_covariance(x, y, mu_x, mu_y)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    C3 = C2 / 2
    return (2 * mu_x * mu_y + C1) * (2 * sig_x * sig_y + C2) * (sig_xy + C3) / (
            (mu_x ** 2 + mu_y ** 2 + C1) * (sig_x ** 2 + sig_y ** 2 + C2) * (sig_x * sig_y + C3))


def correlation_coefficients(x, y):
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    return np.sum((x - mu_x) * (y - mu_y)) / np.sqrt(np.sum((x - mu_x) ** 2) * np.sum((y - mu_y) ** 2))


# 相关系数
def CC(ir, vi, fu):
    rx = correlation_coefficients(ir, fu)
    ry = correlation_coefficients(vi, fu)
    return (rx + ry) / 2


# 空间频率
def SF(I):
    I = I.astype(np.int16)
    RF = np.diff(I, 1, 0)
    RF[RF < 0] = 0
    RF = RF ** 2
    RF[RF > 255] = 255
    RF = np.sqrt(np.mean(RF))

    CF = np.diff(I, 1, 1)
    CF[CF < 0] = 0
    CF = CF ** 2
    CF[CF > 255] = 255
    CF = np.sqrt(np.mean(CF))
    return np.sqrt(RF ** 2 + CF ** 2)


if __name__ == '__main__':
    ir = cv2.imread('test_result/fusion1_gray_l1.png')
    vi = cv2.imread('test_result/fusion1_gray_l1.png')
    fuse_average = cv2.imread('fuse_result/result.jpg')
    fuse = cv2.imread('test_result/fusion1_gray_l1.png')
    # 列名
    col = ['EN', 'SD', 'CC', 'SF']

    # 行名
    row = ['average', 'our']

    # 表格里面的具体值
    vals = np.array([[EN(fuse_average),
                      SD(fuse_average),
                      CC(ir, vi, cv2.imread('fuse_result/result.jpg')),
                      SF(fuse_average)],
                     [EN(fuse),
                      SD(fuse),
                      CC(ir, vi, cv2.imread('test_result/fusion1_gray_l1.png')),
                      SF(fuse)]])

    # 绘制表格
    plt.figure(figsize=(20, 8))
    tab = plt.table(cellText=vals, colLabels=col, rowLabels=row, loc='center', cellLoc='center', rowLoc='center')
    tab.scale(1, 2)
    plt.axis('off')
    plt.show()

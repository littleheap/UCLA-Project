from functools import reduce
import cv2
import matplotlib.pyplot as plt
import time
from scipy.io import *
import spectral
from spectral import *
import joblib
from sklearn.model_selection import *
import numpy as np
from sklearn.svm import SVC
from sklearn import *
import pandas as pd

'''
    在SVM_1800.py分类的基础上，将错误的分类像素点标记出来，实现可视化
'''

input_image = loadmat('../dataset/origin/PaviaU.mat')['paviaU']
output_image = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']

# 导入数据集切割训练与测试数据
data = pd.read_csv('../dataset/PaviaU_gt_band_label_loc_.csv', header=None)
data = data.as_matrix()

# 获取特征矩阵
data_content = data[:, :-2]
# 获取标记矩阵
data_label = data[:, -2]
# 获取位置矩阵
data_loc = data[:, -1]


# 计算正确率
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc = ('%.4f' % np.mean(acc))
    print(tip + '正确率：', (float(acc) * 100), '%')


# 返回错误坐标
def return_wrong(a, b):
    acc = a.ravel() == b.ravel()
    wrong = list()
    i = 0
    for cur in acc:
        if cur:
            pass
        else:
            cur_loc = data_loc[i]
            wrong.append(cur_loc)
        i = i + 1
    return wrong


'''
    SVC+CV
'''
time_start = time.time()

# 加载模型
model = joblib.load('./models/SVC_1800.m')

# 预测gt全集
pred = model.predict(data_content)

# 计算gt全集正确率
show_accuracy(pred, data_label, 'SVC')  # SVC正确率： 90 %

# 获取gt错误分类坐标
wrong = return_wrong(pred, data_label)

# print(wrong)
print(len(wrong))  # 1264

time_end = time.time()
print('totally cost', time_end - time_start)  # 24s

'''
    单颜色画图并将错误的分类像素标记SVM_WrongClass.png
'''

# 读取标记图片paviaU_gt
output_image = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']  # (610, 340)

# 单颜色（黄色）显示标记图片

# 初始化个通道，用于生成新的paviaU_gt
c1 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']
c2 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']
c3 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']

# 现将全部分类坐标用黄色标记
for i in range(610):
    for j in range(340):
        if (output_image[i][j] == 0):
            c1[i][j] = 255
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 1):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 2):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 3):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 4):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 5):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 6):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 7):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 8):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 9):
            c1[i][j] = 0
            c2[i][j] = 255
            c3[i][j] = 255

# 将错误分类坐标用红色标记
for value in wrong:
    i = int(value // 340)
    j = int(value % 340)
    c1[i][j] = 255
    c2[i][j] = 0
    c3[i][j] = 255

# # 合并三个通道，组成三通道RGB图片
# single_merged = cv2.merge([c1, c2, c3])
# # 存储图片
# cv2.imwrite('../imgs/SVM_1800_WrongClass.png', single_merged)
# # 显示图片
# cv2.imshow("output", single_merged)
# # 不闪退
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
    多颜色画图并将错误的分类像素标记SVM_WrongClass_multicolor.png
'''
# 导入gt数据集
pavia_gt = pd.read_csv('../dataset/PaviaU_gt_band_label_loc_.csv', header=None)
pavia_gt = pavia_gt.as_matrix()

# 获取特征矩阵
pavia_gt_content = pavia_gt[:, :-2]
# 获取标记矩阵
pavia_gt_label = pavia_gt[:, -2]
# 获取位置矩阵
pavia_gt_loc = pavia_gt[:, -1]

# 初始化个通道，用于生成新的paviaU_gt
c1 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']
c2 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']
c3 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']

cursor = 0

# 现将全部分类坐标用9种彩色标记，0背景类别用白色
for i in range(610):
    for j in range(340):
        if (output_image[i][j] == 0):
            c1[i][j] = 255
            c2[i][j] = 255
            c3[i][j] = 255
            continue
        cur_band = pavia_gt_content[cursor]
        cur_pred = int(model.predict([cur_band]))
        if (cur_pred == 1):
            c1[i][j] = 20
            c2[i][j] = 104
            c3[i][j] = 82
        if (cur_pred == 2):
            c1[i][j] = 40
            c2[i][j] = 200
            c3[i][j] = 160
        if (cur_pred == 3):
            c1[i][j] = 60
            c2[i][j] = 240
            c3[i][j] = 111
        if (cur_pred == 4):
            c1[i][j] = 80
            c2[i][j] = 77
            c3[i][j] = 190
        if (cur_pred == 5):
            c1[i][j] = 14
            c2[i][j] = 80
            c3[i][j] = 90
        if (cur_pred == 6):
            c1[i][j] = 120
            c2[i][j] = 60
            c3[i][j] = 150
        if (cur_pred == 7):
            c1[i][j] = 140
            c2[i][j] = 200
            c3[i][j] = 255
        if (cur_pred == 8):
            c1[i][j] = 160
            c2[i][j] = 5
            c3[i][j] = 100
        if (cur_pred == 9):
            c1[i][j] = 180
            c2[i][j] = 180
            c3[i][j] = 255
        cursor = cursor + 1

# 将错误分类坐标用黑色标记
for value in wrong:
    j = int(value % 340)
    i = int(value // 340)
    c1[i][j] = 0
    c2[i][j] = 0
    c3[i][j] = 0

# 合并三个通道，组成三通道RGB图片
single_merged = cv2.merge([c1, c2, c3])
# 存储图片
cv2.imwrite('../imgs/SVM_1800_WrongClass_multicolor.png', single_merged)
# 显示图片
cv2.imshow("output", single_merged)
# 不闪退
cv2.waitKey(0)
cv2.destroyAllWindows()

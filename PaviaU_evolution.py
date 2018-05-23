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
    PaviaU.py的进化程序，根据PaviaU_draw.py绘图情况，
    考虑用围棋吃子的思想，如果某一像素的上下左右全部为一个分类，则将该像素重新协同分类
'''

'''
    训练模型并存储模型
'''
# 导入数据集切割训练与测试数据
data = pd.read_csv('./dataset/PaviaU_draw_wrong.csv', header=None)
data = data.as_matrix()

# 获取特征矩阵
data_content = data[:, :-2]

# 获取标记矩阵
data_label = data[:, -2]

# 获取位置矩阵
data_loc = data[:, -1]

# 切割训练集测试集
data_train, data_test, label_train, label_test = train_test_split(data_content, data_label, test_size=0.3)


# print(data_train.shape)  # (29943, 103)
# print(data_test.shape)  # (12833, 103)
# print(label_train.shape)  # (29943,)
# print(label_test.shape)  # (12833,)

# 计算正确率
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc = ('%.4f' % np.mean(acc))
    print(tip + '正确率：', (float(acc) * 100), '%')


# 返回错误坐标
def return_wrong(a, b, tip):
    acc = a.ravel() == b.ravel()
    # print(acc.shape)  # (42776,)
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


# 返回给定坐标的预测值
def return_class(i, j, pred):
    return pred[i * 340 + j]


'''
    SVC+CV
'''
time_start = time.time()

# 加载模型
model = joblib.load('./models/PaviaU/SVC_CV.m')

# 预测全集
pred = model.predict(data_content)
# print(pred)  # [3. 1. 1. ... 2. 2. 2.]
# print(pred.shape)  # (42776,)

# 计算全集精度
show_accuracy(pred, data_label, 'SVC_CV')

# 查找错误分类坐标
wrong = return_wrong(pred, data_label, 'SVC_CV')
# print(wrong)
# print(len(wrong))  # 1627

# input_image = loadmat('./dataset/PaviaU.mat')['paviaU']

# # 初始化通道列表
# band = []
#
# # 提取有类别标记的通道，并统计计入bandwithlabel
# for i in range(610):
#     for j in range(340):
#         currentband = list(input_image[i][j])
#         band.append(currentband)
#
# # 转化为矩阵
# band = np.array(band)
# print(band.shape)  # (207400, 103)
# band = preprocessing.StandardScaler().fit_transform(band[:, :])
# band = pd.DataFrame(band)
# band.to_csv('./dataset/PaviaU_all.csv', header=False, index=False)

# 全部像素207400
band = pd.read_csv('./dataset/PaviaU_all.csv', header=None)
band = band.as_matrix()

# test = model.predict([band[1, :]])
# print(test)

# 有标记像素42776
data = pd.read_csv('./dataset/PaviaU_draw_wrong.csv', header=None)
data = data.as_matrix()

# 获取特征矩阵
data_content = data[:, :-2]

# 获取标记矩阵
data_label = data[:, -2]

# 获取位置矩阵
data_loc = data[:, -1]

# 位置-类别字典
dic_loc = dict()
for cur in data:
    dic_loc[int(cur[-1])] = int(cur[-2])
# print(len(dic_loc))  # 42776
print(dic_loc)

change = 0

change_list = list()

for value in wrong:
    # 获取当前预测错误的像素坐标
    i = int(value // 340)
    j = int(value % 340)
    # up
    i_up = i - 1
    j_up = j
    # down
    i_down = i + 1
    j_down = j
    # left
    i_left = i
    j_left = j - 1
    # right
    i_right = i
    j_right = j + 1
    if i_up < 0 or i_down > 610 or j_left < 0 or j_right > 340:
        continue
    # 统计上下左右四个类别
    up_band = band[i_up * 340 + j_up, :]
    up_pred = model.predict([up_band])
    down_band = band[i_down * 340 + j_down, :]
    down_pred = model.predict([down_band])
    left_band = band[i_left * 340 + j_left, :]
    left_pred = model.predict([left_band])
    right_band = band[i_right * 340 + j_right, :]
    right_pred = model.predict([right_band])
    if up_pred == down_pred == left_pred == right_pred:
        cur_band = band[int(value), :]
        cur_pred = model.predict([cur_band])
        # print(cur_pred)
        cur_label = data_loc[int(value)]
        # print(cur_label)
        if cur_pred == cur_label:
            change = change + 1
            change_list.append(int(value))

# 查找错误分类坐标
wrong = return_wrong(pred, data_label, 'SVC_CV')
# print(wrong)
# print(len(wrong))  # 1627

print('新正确率：', (42776 - 1627 + change) / 42776)

time_end = time.time()
print('total time：', time_end - time_start)

'''
    #######
    开始画图
    #######
'''
# 读取标记图片paviaU_gt
output_image = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']  # (610, 340)

# 单颜色（黄色）显示标记图片

# 初始化个通道，用于生成新的paviaU_gt
c1 = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']

c2 = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']

c3 = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']

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
    j = int(value % 340)
    i = int(value // 340)
    c1[i][j] = 255
    c2[i][j] = 0
    c3[i][j] = 255

# 将更正的分类坐标重新用黄色标记
for value in change_list:
    j = int(value % 340)
    i = int(value // 340)
    c1[i][j] = 0
    c2[i][j] = 255
    c3[i][j] = 255

# 合并三个通道，组成三通道RGB图片
single_merged = cv2.merge([c1, c2, c3])

# 显示图片
cv2.imshow("output", single_merged)

# 不闪退
cv2.waitKey(0)
cv2.destroyAllWindows()

# 存储图片
cv2.imwrite('./images/paviaU_gt_evolution.png', single_merged)

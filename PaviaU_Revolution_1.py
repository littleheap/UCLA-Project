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
    PaviaU.py的进化程序，根据PaviaU_WrongClass.py绘图情况，
    考虑用围棋吃子的思想，如果某一像素的上下左右全部为一个分类，则将该像素重新协同分类
'''
# 读取原始图片paviaU
input_image = loadmat('./dataset/PaviaU.mat')['paviaU']

# 读取标记图片paviaU_gt
output_image = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']  # (610, 340)

# 导入gt数据集
pavia_gt = pd.read_csv('./dataset/PaviaU_gt_band_label_loc.csv', header=None)
pavia_gt = pavia_gt.as_matrix()
# 获取特征矩阵
pavia_gt_content = pavia_gt[:, :-2]
# 获取标记矩阵
pavia_gt_label = pavia_gt[:, -2]
# 获取位置矩阵
pavia_gt_loc = pavia_gt[:, -1]

# 导入全部数据集
pavia = pd.read_csv('./dataset/PaviaU_band_label_loc.csv', header=None)
pavia = pavia.as_matrix()
# 获取通道矩阵
pavia_content = pavia[:, :-2]
# 获取标记矩阵
pavia_label = pavia[:, -2]
# 获取位置矩阵
pavia_loc = pavia[:, -1]


# 计算正确率
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc = ('%.4f' % np.mean(acc))
    print(tip + '正确率：', (float(acc) * 100), '%')


# 返回错误坐标
def return_wrong(a, b):
    acc = a.ravel() == b.ravel()
    # print(acc.shape)  # (42776,)
    wrong = list()
    i = 0
    for cur in acc:
        if cur:
            pass
        else:
            cur_loc = pavia_gt_loc[i]
            wrong.append(cur_loc)
        i = i + 1
    return wrong


'''
    SVC+CV
'''
time_start = time.time()

# 加载模型
model = joblib.load('./models/PaviaU/SVC_CV.m')

# 预测gt全集
pred = model.predict(pavia_gt_content)
# print(pred)  # [3. 1. 1. ... 2. 2. 2.]
# print(pred.shape)  # (42776,)

# 计算gt全集精度
show_accuracy(pred, pavia_gt_label, 'SVC_CV')

# 查找gt错误分类坐标
wrong_loc = return_wrong(pred, pavia_gt_label)
# print(wrong_loc)
# print(len(wrong_loc))  # 1627

# gt{位置:类别}字典
dic_loc = dict()
for cur in pavia_gt:
    dic_loc[int(cur[-1])] = int(cur[-2])
# print(len(dic_loc))  # 42776
# print(dic_loc)

# gt{位置:通道}字典
dic_band = dict()
for cur in pavia_gt:
    dic_band[int(cur[-1])] = cur[:-2]
print(len(dic_band))  # 42776
# print(dic_band)

# 修正错误分类过程中，成功更改的像素
change_num = 0

change_list = list()

for loc in wrong_loc:
    # 获取当前预测错误的像素坐标
    i = int(loc // 340)
    j = int(loc % 340)
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
    if i_up < 0 or i_down > 609 or j_left < 0 or j_right > 339:
        continue
    # 统计上下左右四个类别
    up_loc = i_up * 340 + j_up
    down_loc = i_down * 340 + j_down
    left_loc = i_left * 340 + j_left
    right_loc = i_right * 340 + j_right
    # 如果当前错误的像素，对应的上下左右四个像素有一个不在gt标记范围内，就跳过，这说明该像素在边缘
    if up_loc not in pavia_gt_loc or down_loc not in pavia_gt_loc or left_loc not in pavia_gt_loc or right_loc not in pavia_gt_loc:
        continue
    # 如果当前错误的像素完全包裹在gt标记范围内部
    # （1）先获取上下左右四个像素的对应通道
    # （2）对四个像素做预测，如果预测一致属于同一类，则当前错误像素与上下左右四个像素同化
    up_band = dic_band[up_loc]
    up_pred = model.predict([up_band])
    down_band = dic_band[down_loc]
    down_pred = model.predict([down_band])
    left_band = dic_band[left_loc]
    left_pred = model.predict([left_band])
    right_band = dic_band[right_loc]
    right_pred = model.predict([right_band])
    # print([up_pred, down_pred, left_pred, right_pred], [up_pred == down_pred == left_pred == right_pred])
    # 当预测错误的坐标上下左右预测都一致时，将此坐标的原有预测值更新为上下左右一致对的预测值
    if up_pred == down_pred == left_pred == right_pred:
        cur_pred = int(up_pred)
        cur_label = pavia_label[int(loc)]
        # print([cur_pred, cur_label])
        # 如果通过该方式修正的像素确实修正正确，那么记录到修正的记录中
        if cur_pred == cur_label:
            change_num = change_num + 1
            change_list.append(int(loc))

new_acc = '%.4f' % ((42776 - len(wrong_loc) + change_num) / 42776)
print('新正确率：', float(new_acc) * 100, '%')  # 0.9700（提升了1%）
print(change_num)  # 342
print(change_list)

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
for value in wrong_loc:
    j = int(value % 340)
    i = int(value // 340)
    c1[i][j] = 255
    c2[i][j] = 0
    c3[i][j] = 255

# 将更正的分类坐标重新用蓝色标记
for value in change_list:
    j = int(value % 340)
    i = int(value // 340)
    c1[i][j] = 255
    c2[i][j] = 255
    c3[i][j] = 0

# 合并三个通道，组成三通道RGB图片
single_merged = cv2.merge([c1, c2, c3])

# 显示图片
cv2.imshow("output", single_merged)

# 不闪退
cv2.waitKey(0)
cv2.destroyAllWindows()

# 存储图片
cv2.imwrite('./images/paviaU_gt_Rvolution.png', single_merged)

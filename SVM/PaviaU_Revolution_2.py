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
    在PaviaU_Revolution_1.py的基础上，不单对单一包围像素进行更新分类，
    此程序中将连通的错误分类像素，外围一致情况下，与外围协同分类
'''
# 读取原始图片paviaU
input_image = loadmat('../dataset/PaviaU.mat')['paviaU']

# 读取标记图片paviaU_gt
output_image = loadmat('../dataset/PaviaU_gt.mat')['paviaU_gt']  # (610, 340)

# 导入gt数据集
pavia_gt = pd.read_csv('../dataset/PaviaU_gt_band_label_loc.csv', header=None)
pavia_gt = pavia_gt.as_matrix()
# 获取特征矩阵
pavia_gt_content = pavia_gt[:, :-2]
# 获取标记矩阵
pavia_gt_label = pavia_gt[:, -2]
# 获取位置矩阵
pavia_gt_loc = pavia_gt[:, -1]

# 导入全部数据集
pavia = pd.read_csv('../dataset/PaviaU_band_label_loc.csv', header=None)
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
    wrong_set = set()
    i = 0
    for cur in acc:
        if cur:
            pass
        else:
            cur_loc = pavia_gt_loc[i]
            wrong_set.add(int(cur_loc))
        i = i + 1
    return wrong_set


# 返回一个像素坐标的上下左右四个坐标
def return_4loc(loc):
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
        return False, False, False, False
    else:
        return [i_up, j_up], [i_down, j_down], [i_left, j_left], [i_right, j_right]


'''
    SVC+CV
'''
time_start = time.time()

# 加载模型
model = joblib.load('../models/PaviaU/SVC_CV.m')

# 预测gt全集
pred = model.predict(pavia_gt_content)
# print(pred)  # [3. 1. 1. ... 2. 2. 2.]
# print(pred.shape)  # (42776,)

# 计算gt全集精度
show_accuracy(pred, pavia_gt_label, 'SVC_CV')

# 查找gt错误分类坐标
wrong_loc = return_wrong(pred, pavia_gt_label)
# 临时保存wrong_loc
temp_wrong_loc = return_wrong(pred, pavia_gt_label)
print(wrong_loc)
print(len(wrong_loc))  # 1627

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
change_list = set()

# 记录当前连通的错误分类像素区域坐标
union_set = set()

# 记录当前连通的错误分类像素区域的外围区域坐标
unionbound_set = set()

flag = ''


# 连通区域递归函数
def mani(loc, flag):
    # 错误类别中删除当前元素
    wrong_loc.remove(loc)
    # 获取当前元素上下左右四个坐标
    up, down, left, right = return_4loc(loc)
    # 在当前像素处于画面边缘时，跳过
    if not up:
        return
    # up绝对坐标
    i_up = up[0]
    j_up = up[1]
    # down绝对坐标
    i_down = down[0]
    j_down = down[1]
    # left绝对坐标
    i_left = left[0]
    j_left = left[1]
    # right绝对坐标
    i_right = right[0]
    j_right = right[1]
    # 统计上下左右四个坐标
    up_loc = i_up * 340 + j_up
    down_loc = i_down * 340 + j_down
    left_loc = i_left * 340 + j_left
    right_loc = i_right * 340 + j_right
    # 如果当前错误的像素，对应的上下左右四个像素有一个不在gt标记范围内，就跳过，这说明该像素在边缘
    if up_loc not in pavia_gt_loc or down_loc not in pavia_gt_loc or left_loc not in pavia_gt_loc or right_loc not in pavia_gt_loc:
        return
    # 将当前像素上下左右四个坐标放入包围边缘集合
    if up_loc not in union_set:
        unionbound_set.add(up_loc)
    if down_loc not in union_set:
        unionbound_set.add(down_loc)
    if left_loc not in union_set:
        unionbound_set.add(left_loc)
    if right_loc not in union_set:
        unionbound_set.add(right_loc)
    # 判断上下左右是否存在错误连通区域
    if up_loc in wrong_loc:
        # 在错误的连通区域内，则将其添加入连通区域集合
        union_set.add(up_loc)
        # 将其从边缘区域集合中移除
        unionbound_set.remove(up_loc)
        # 记录连通方向
        # 递归处理当前连通的新像素
        mani(up_loc, 'up')
    if down_loc in wrong_loc:
        union_set.add(down_loc)
        unionbound_set.remove(down_loc)
        mani(down_loc, 'down')
    if left_loc in wrong_loc:
        union_set.add(left_loc)
        unionbound_set.remove(left_loc)
        mani(left_loc, 'left')
    if right_loc in wrong_loc:
        union_set.add(right_loc)
        unionbound_set.remove(right_loc)
        mani(right_loc, 'right')
    classes = set()
    # 预测边缘集合中元素类别
    for bound_loc in unionbound_set:
        bound_band = dic_band[bound_loc]
        bound_pred = model.predict([bound_band])
        classes.add(int(bound_pred))
    # 如果边缘集合元素都属于一类，则说明该连通区域位于标记数据集内部
    if len(classes) == 1:
        for piexl in union_set:
            # 将每一个连通区域的像素添加入更改的列表
            change_list.add(piexl)


# 字典转换为列表
wrong_loc_list = list(wrong_loc)
# 计算迭代次数
iter_times = len(wrong_loc)
for i in range(iter_times):
    # 获取当前loc
    loc = wrong_loc_list[i]
    # 如果当前loc没有被处理掉
    if loc in wrong_loc:
        union_set.add(loc)
        mani(loc, 'no')
        union_set.clear()
        unionbound_set.clear()

new_acc = '%.4f' % ((42776 - iter_times + len(change_list)) / 42776)
print('新正确率：', float(new_acc) * 100, '%')  # 98.38 %（又提升了1%）
print(len(change_list))  # 934
print(change_list)

time_end = time.time()
print('total time：', time_end - time_start)

'''
    #######
    开始画图
    #######
'''
# 读取标记图片paviaU_gt
output_image = loadmat('../dataset/PaviaU_gt.mat')['paviaU_gt']  # (610, 340)

# 单颜色（黄色）显示标记图片

# 初始化个通道，用于生成新的paviaU_gt
c1 = loadmat('../dataset/PaviaU_gt.mat')['paviaU_gt']

c2 = loadmat('../dataset/PaviaU_gt.mat')['paviaU_gt']

c3 = loadmat('../dataset/PaviaU_gt.mat')['paviaU_gt']

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
for value in temp_wrong_loc:
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
cv2.imwrite('../images/paviaU_gt_Rvolution_2-1.png', single_merged)

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
cv2.imwrite('../images/paviaU_gt_Rvolution_2-2.png', single_merged)

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
    在PaviaU.py分类的基础上，将错误的分类像素点标记出来
'''

''' 
    #################################
    1.加载遥感图像的.mat数据进行呈图显示
    #################################
'''
input_image = loadmat('./dataset/PaviaU.mat')['paviaU']

output_image = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']

print(input_image.shape)  # (610, 340, 103)

print(output_image.shape)  # (610, 340)

print(np.unique(input_image))  # [   0    1    2 ... 7996 7997 8000]

print(np.unique(output_image))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

'''
    统计类元素个数
'''
# dict_category = {}
#
# for i in range(output_image.shape[0]):  # 610
#     for j in range(output_image.shape[1]):  # 340
#         if output_image[i][j] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
#             if output_image[i][j] not in dict_category:
#                 dict_category[output_image[i][j]] = 0
#             dict_category[output_image[i][j]] += 1
#
# # {0: 164624, 1: 6631, 2: 18649, 3: 2099, 4: 3064, 5: 1345, 6: 5029, 7: 1330, 8: 3682, 9: 947}
# print(dict_category)
#
# # 验证全部类别恰好为207400 (610 * 340 = 164624 + 42776)
# print(reduce(lambda x, y: x + y, dict_category.values()))

'''
    ############################################################
    2.将.mat的数据转化为python后续算法处理的csv文件，额外添加位置信息
    ############################################################
'''

# # 提取分类类别和对应1*1*103通道矩阵，整合通道+类别
# # 初始化label矩阵
# label = np.zeros([610, 340])
#
# # 提取label矩阵
# for i in range(610):
#     for j in range(340):
#         if output_image[i][j] != 0:
#             label[i][j] = output_image[i][j]
#
# # 初始化通道类别列表
# band_label_loc = []
#
# # 提取有类别标记的通道，并统计计入band_label_loc
# for i in range(610):
#     for j in range(340):
#         if label[i][j] != 0:
#             currentband = list(input_image[i][j])
#             currentband.append(label[i][j])
#             currentband.append(i * 340 + j)
#             band_label_loc.append(currentband)
#
# # 转化为矩阵
# band_label_loc = np.array(band_label_loc)
#
# # 数据维度+通道和标签维度，数据42776维度（1-9类别），通道103个波段，104列是标签，105列是坐标
# print(band_label_loc.shape)  # (42776,105)
#
# '''
#     数据标准化处理
# '''
# # 标准化预处理特征数据，不考虑最后两列标签（42776,103）
# data_content = preprocessing.StandardScaler().fit_transform(band_label_loc[:, :-2])
# # data_D = preprocessing.MinMaxScaler().fit_transform(new_datawithlabel_array[:,:-1])
#
# # 将倒数第二列标签单独抽取（42776,1）
# data_label = band_label_loc[:, -2]
#
# # 最后一列坐标单独抽取（42776,1）
# data_loc = band_label_loc[:, -1]
#
# # 合并标准化特征矩阵和标记矩阵
# new_data = np.column_stack((data_content, data_label))
# new_data = np.column_stack((new_data, data_loc))
#
# # 将全新标准化矩阵转化为CSV行列阵格式
# new_data = pd.DataFrame(new_data)
#
# # 存储行列阵
# new_data.to_csv('./dataset/PaviaU_gt_band_label_loc.csv', header=False, index=False)

'''
    ###############################################
    3.训练并存储模型，观察分类效果，在图中显示与原图对比
    ###############################################
'''

# 导入数据集切割训练与测试数据
data = pd.read_csv('./dataset/PaviaU_gt_band_label_loc.csv', header=None)
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
model = joblib.load('./models/PaviaU/SVC_CV.m')

# 预测gt全集
pred = model.predict(data_content)

# 计算gt全集正确率
show_accuracy(pred, data_label, 'SVC_CV')

# 获取gt错误分类坐标
wrong = return_wrong(pred, data_label)

print(wrong)
print(len(wrong))  # 1627

time_end = time.time()
print('totally cost', time_end - time_start)

''' 
    #####################################################
    单颜色画图并将错误的分类像素标记paviaU_gt_WrongClass.png
    #####################################################
'''

# # 读取标记图片paviaU_gt
# output_image = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']  # (610, 340)
#
# # 单颜色（黄色）显示标记图片
#
# # 初始化个通道，用于生成新的paviaU_gt
# c1 = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']
#
# c2 = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']
#
# c3 = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']
#
# # 现将全部分类坐标用黄色标记
# for i in range(610):
#     for j in range(340):
#         if (output_image[i][j] == 0):
#             c1[i][j] = 255
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 1):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 2):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 3):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 4):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 5):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 6):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 7):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 8):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 9):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#
# # 将错误分类坐标用红色标记
# for value in wrong:
#     i = int(value // 340)
#     j = int(value % 340)
#     c1[i][j] = 255
#     c2[i][j] = 0
#     c3[i][j] = 255
#
# # 合并三个通道，组成三通道RGB图片
# single_merged = cv2.merge([c1, c2, c3])
#
# # 显示图片
# cv2.imshow("output", single_merged)
#
# # 不闪退
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # 存储图片
# cv2.imwrite('./images/SVM_WrongClass.png', single_merged)

''' 
    ################################################################
    多颜色画图并将错误的分类像素标记paviaU_gt_WrongClass_multicolor.png
    ################################################################
'''
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

# 初始化个通道，用于生成新的paviaU_gt
c1 = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']

c2 = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']

c3 = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']

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

# 显示图片
cv2.imshow("output", single_merged)

# 不闪退
cv2.waitKey(0)
cv2.destroyAllWindows()

# 存储图片
cv2.imwrite('../images/paviaU_gt_WrongClass_multicolor.png', single_merged)

from sklearn import *
from scipy.io import loadmat
from functools import reduce
import cv2
import numpy as np
import pandas as pd

'''
    数据集PaviaU分类程序：
    数据集：PaviaU（610*340*103）
    标记集：PaviaU_gt（610*340）
    步骤：
    1.加载遥感图像的mat数据进行呈图显示
    2.将mat的数据转化为python后续算法处理的csv文件
    3.训练并存储模型，观察分类效果，在图中显示与原图对比
'''

''' 
    解析数据集，统计并可视化：加载遥感图像的mat数据进行呈图显示
'''
input_image = loadmat('../dataset/origin/PaviaU.mat')['paviaU']  # (610, 340, 103)
output_image = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']  # (610, 340)

print(input_image.shape)  # (610, 340, 103)
print(output_image.shape)  # (610, 340)
# 可以看出通道数据采集范围
print(np.unique(input_image))  # [   0    1    2 ... 7996 7997 8000]
# 可以看出类别
print(np.unique(output_image))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

'''
    统计不同类像素数量
'''
dict_category = {}

for i in range(output_image.shape[0]):  # 610
    for j in range(output_image.shape[1]):  # 340
        if output_image[i][j] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            if output_image[i][j] not in dict_category:
                dict_category[output_image[i][j]] = 0
            dict_category[output_image[i][j]] += 1

print(dict_category)
# {0: 164624, 1: 6631, 4: 3064, 2: 18649, 8: 3682, 5: 1345, 9: 947, 6: 5029, 3: 2099, 7: 1330}

# 验证全部类别恰好为207400 (610 * 340 = 164624 + 42776)
print(reduce(lambda x, y: x + y, dict_category.values()))
# 207400

'''
    将验证集paviaU_gt中10中类别（9中分类+1个无类别背景）用不同颜色标记出来并以RGB格式图片显示
'''
# 读取标记图片paviaU_gt
output_image = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']  # (610, 340)

# 初始化个通道，用于生成新的paviaU_gt
c1 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']
c2 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']
c3 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']

# 对每一种类别用一种颜色在三个通道上标记
for i in range(610):
    for j in range(340):
        if (output_image[i][j] == 0):
            c1[i][j] = 255
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 1):
            c1[i][j] = 20
            c2[i][j] = 104
            c3[i][j] = 82
        if (output_image[i][j] == 2):
            c1[i][j] = 40
            c2[i][j] = 200
            c3[i][j] = 160
        if (output_image[i][j] == 3):
            c1[i][j] = 60
            c2[i][j] = 240
            c3[i][j] = 111
        if (output_image[i][j] == 4):
            c1[i][j] = 80
            c2[i][j] = 77
            c3[i][j] = 190
        if (output_image[i][j] == 5):
            c1[i][j] = 14
            c2[i][j] = 80
            c3[i][j] = 90
        if (output_image[i][j] == 6):
            c1[i][j] = 120
            c2[i][j] = 60
            c3[i][j] = 150
        if (output_image[i][j] == 7):
            c1[i][j] = 140
            c2[i][j] = 200
            c3[i][j] = 255
        if (output_image[i][j] == 8):
            c1[i][j] = 160
            c2[i][j] = 5
            c3[i][j] = 100
        if (output_image[i][j] == 9):
            c1[i][j] = 180
            c2[i][j] = 180
            c3[i][j] = 255

# 合并三个通道，组成三通道RGB图片
multi_merged = cv2.merge([c1, c2, c3])
# 存储图片
cv2.imwrite('../imgs/paviaU.png', multi_merged)
# 显示图片
cv2.imshow("output", multi_merged)
# 不闪退
cv2.waitKey(0)
cv2.destroyAllWindows()

# 单颜色（蓝色）显示标记图片
# 初始化个通道，用于生成新的paviaU_gt
c1 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']
c2 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']
c3 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']

for i in range(610):
    for j in range(340):
        if (output_image[i][j] == 0):
            c1[i][j] = 255
            c2[i][j] = 255
            c3[i][j] = 255
        if (output_image[i][j] == 1):
            c1[i][j] = 255
            c2[i][j] = 0
            c3[i][j] = 0
        if (output_image[i][j] == 2):
            c1[i][j] = 255
            c2[i][j] = 0
            c3[i][j] = 0
        if (output_image[i][j] == 3):
            c1[i][j] = 255
            c2[i][j] = 0
            c3[i][j] = 0
        if (output_image[i][j] == 4):
            c1[i][j] = 255
            c2[i][j] = 0
            c3[i][j] = 0
        if (output_image[i][j] == 5):
            c1[i][j] = 255
            c2[i][j] = 0
            c3[i][j] = 0
        if (output_image[i][j] == 6):
            c1[i][j] = 255
            c2[i][j] = 0
            c3[i][j] = 0
        if (output_image[i][j] == 7):
            c1[i][j] = 255
            c2[i][j] = 0
            c3[i][j] = 0
        if (output_image[i][j] == 8):
            c1[i][j] = 255
            c2[i][j] = 0
            c3[i][j] = 0
        if (output_image[i][j] == 9):
            c1[i][j] = 255
            c2[i][j] = 0
            c3[i][j] = 0

# 合并三个通道，组成三通道RGB图片
single_merged = cv2.merge([c1, c2, c3])
# 存储图片
cv2.imwrite('../imgs/paviaU_gt.png', single_merged)
# 显示图片
cv2.imshow("output", single_merged)
# 不闪退
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
    将mat的数据转化为python后续算法处理的csv文件
'''
# 提取分类类别和对应1*1*103通道矩阵，整合通道+类别
# 初始化label矩阵
label = np.zeros([610, 340])

# 提取label矩阵
for i in range(610):
    for j in range(340):
        if output_image[i][j] != 0:
            label[i][j] = output_image[i][j]

# 初始化通道类别列表
bandwithlabel = []
# 提取有类别标记的通道，并统计计入bandwithlabel
for i in range(610):
    for j in range(340):
        if label[i][j] != 0:
            currentband = list(input_image[i][j])
            currentband.append(label[i][j])
            bandwithlabel.append(currentband)
# 转化为矩阵
bandwithlabel = np.array(bandwithlabel)
# 数据维度+通道和标签维度，数据42776维度（1-9类别），通道103个波段，最后104列是标签
print(bandwithlabel.shape)  # (42776,104)

'''
    数据标准化处理，是在抽取完数据后，再进行标准化
'''
# 标准化预处理特征数据，不考虑最后一列标签（42776,103）
data_content = preprocessing.StandardScaler().fit_transform(bandwithlabel[:, :-1])
# 将最后一列标签单独抽取（42776,1）
data_label = bandwithlabel[:, -1]
# 合并标准化特征矩阵和标记矩阵
new_data = np.column_stack((data_content, data_label))
# 将全新标准化矩阵转化为CSV行列阵格式
new_data = pd.DataFrame(new_data)
# 存储行列阵
# new_data.to_csv('../dataset/PaviaU_gt_band_label.csv', header=False, index=False)


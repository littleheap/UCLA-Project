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

# 加载模型
model = joblib.load('./models/SVC_1800.m')

# 创建统计矩阵
mat = [[0] * 340] * 610
mat_np = np.array(mat)

print(mat_np.shape)  # (610, 340)

for i in range(42776):
    # 对当前条提取三围数据
    band = data_content[i][np.newaxis, :]  # 数据增维才能用模型预测
    label = data_label[i]
    loc = data_loc[i]
    # 换算二维坐标
    loc_x = int(loc) // 340
    loc_y = int(loc) % 340
    print([loc_x, loc_y, loc])
    # 判定
    pred = model.predict(band)
    mat_np[loc_x][loc_y] = pred

mat_np = pd.DataFrame(mat_np)
mat_np.to_csv('../dataset/SVM_pred.csv', header=False, index=False)
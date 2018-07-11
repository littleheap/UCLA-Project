from functools import reduce
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

# 导入数据集切割训练与测试数据（用最后处理出的最标准的数据，标准化统一）
data = pd.read_csv('../dataset/PaviaU_gt_band_label_loc_.csv', header=None)
data = data.as_matrix()

# 获取特征矩阵
data_content = data[:, :-2]
# print(data_content.shape)  # (42776, 103)

# 获取标记矩阵
data_label = data[:, -2]
# print(data_label.shape)  # (42776,)

# 切割训练集测试集
data_train, data_test, label_train, label_test = train_test_split(data_content, data_label, test_size=0.3)

# print(data_train.shape)  # (29943, 103)
# print(data_test.shape)  # (12833, 103)
# print(label_train.shape)  # (29943,)
# print(label_test.shape)  # (12833,)

'''
    获取9个类别，每个类别200条，一共1800条数据，每条数据103通道信息+一列label，一共(1800,104)
'''


# 获取不同类别指定个数的通道数据，返回该类别二维通道数据，类别label和行数number指定，通道数为103，104通道为label
def get_band(label, n):
    band = []
    for i in range(42776):
        currentband = list(data_content[i])
        currentlabel = data_label[i]
        currentband.append(label)
        if currentlabel == label:
            band.append(currentband)
        if len(list(band)) == n:
            break
    band = list(band)
    # print(len(band))
    band_matrix = np.array(band)
    # print(band.shape)
    # print(band_matrix)
    return band_matrix


data = list()

for i in range(10):
    if i == 0:
        continue
    band_i = get_band(i, 200)
    for j in range(200):
        row = band_i[j, :]
        data.append(row)

new_data = pd.DataFrame(data)
print(new_data.shape)
new_data.to_csv('./dataset/CNN_data.csv', header=False, index=False)

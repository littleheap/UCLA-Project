# 读取原始图片paviaU
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import preprocessing

input_image = loadmat('./dataset/PaviaU.mat')['paviaU']
output_image = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']  # (610, 340)

'''
    PaviaU_band.csv
'''
# band = []
# for i in range(610):
#     for j in range(340):
#         currentband = list(input_image[i][j])
#         band.append(currentband)
# band = np.array(band)
# print(band.shape)  # (207400, 103)
# band = preprocessing.StandardScaler().fit_transform(band[:, :])
# band = pd.DataFrame(band)
# band.to_csv('./dataset/PaviaU_band.csv', header=False, index=False)

'''
    PaviaU_band_label_loc.csv
'''
# label = np.zeros([610, 340])
# for i in range(610):
#     for j in range(340):
#         if output_image[i][j] != 0:
#             label[i][j] = output_image[i][j]
#
# band_label_loc = []
# for i in range(610):
#     for j in range(340):
#         currentband = list(input_image[i][j])
#         currentband.append(label[i][j])
#         currentband.append(i * 340 + j)
#         band_label_loc.append(currentband)
# band_label_loc = np.array(band_label_loc)
# print(band_label_loc.shape)  # (207400,105)
#
# data_content = preprocessing.StandardScaler().fit_transform(band_label_loc[:, :-2])
# data_label = band_label_loc[:, -2]
# data_loc = band_label_loc[:, -1]
#
# new_data = np.column_stack((data_content, data_label))
# new_data = np.column_stack((new_data, data_loc))
# new_data = pd.DataFrame(new_data)
# new_data.to_csv('./dataset/PaviaU_band_label_loc.csv', header=False, index=False)

'''
    PaviaU_gt_band_label_loc
'''
# label = np.zeros([610, 340])
# for i in range(610):
#     for j in range(340):
#         if output_image[i][j] != 0:
#             label[i][j] = output_image[i][j]
#
# band_label_loc = []
# for i in range(610):
#     for j in range(340):
#         if label[i][j] != 0:
#             currentband = list(input_image[i][j])
#             currentband.append(label[i][j])
#             currentband.append(i * 340 + j)
#             band_label_loc.append(currentband)
#
# band_label_loc = np.array(band_label_loc)
# print(band_label_loc.shape)  # (42776,105)
#
# data_content = preprocessing.StandardScaler().fit_transform(band_label_loc[:, :-2])
# data_label = band_label_loc[:, -2]
# data_loc = band_label_loc[:, -1]
#
# new_data = np.column_stack((data_content, data_label))
# new_data = np.column_stack((new_data, data_loc))
# new_data = pd.DataFrame(new_data)
#
# new_data.to_csv('./dataset/PaviaU_gt_band_label_loc.csv', header=False, index=False)

'''
    PaviaU_gt_band_label_loc_：与上面的不同，此数据的标准化是在全部数据标准化后抽取出来的
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
print(pavia_gt_loc)

# 导入全部数据集
pavia = pd.read_csv('./dataset/PaviaU_band_label_loc.csv', header=None)
pavia = pavia.as_matrix()
# 获取通道矩阵
pavia_content = pavia[:, :-2]
# 获取标记矩阵
pavia_label = pavia[:, -2]
# 获取位置矩阵
pavia_loc = pavia[:, -1]

i = 0

data = list()

for row in pavia:
    row_loc = int(row[-1])
    if(row_loc in pavia_gt_loc):
        data.append(row)
        i = i + 1

new_data = pd.DataFrame(data)
print(new_data.shape)
new_data.to_csv('./dataset/PaviaU_gt_band_label_loc_.csv', header=False, index=False)

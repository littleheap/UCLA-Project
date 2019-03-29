import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import preprocessing

# 读取原始图片paviaU
input_image = loadmat('../dataset/origin/PaviaU.mat')['paviaU']  # (610, 340, 103)
output_image = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']  # (610, 340)

'''
    PaviaU_band.csv：将(610, 340, 103)原始三维矩阵转化为二维矩阵(207400, 103)
'''
band = []
for i in range(610):
    for j in range(340):
        currentband = list(input_image[i][j])
        band.append(currentband)
band = np.array(band)
print(band.shape)  # (207400, 103)
# 标准化处理：每列转化为均值为0，标准差为1的正态分布，避免异常值问题
band = preprocessing.StandardScaler().fit_transform(band[:, :])
# 转换成dataframe数据格式
band = pd.DataFrame(band)
# 生成(207400, 103)像素通道信息矩阵
band.to_csv('../dataset/PaviaU_band.csv', header=False, index=False)

'''
    PaviaU_band_label_loc.csv：比PaviaU_band.csv尾部附加上label和像素位置序号两个信息
'''
# 整理label矩阵
label = np.zeros([610, 340])
for i in range(610):
    for j in range(340):
        if output_image[i][j] != 0:
            label[i][j] = output_image[i][j]

# 整理"通道-label-位置"信息矩阵
band_label_loc = []
for i in range(610):
    for j in range(340):
        # 获取当前像素通道信息
        currentband = list(input_image[i][j])
        # 附加对应label
        currentband.append(label[i][j])
        # 附加对应位置：位置采用计算行列位置编号方式
        currentband.append(i * 340 + j)
        # 添加到总矩阵信息
        band_label_loc.append(currentband)
band_label_loc = np.array(band_label_loc)
# 验证数据格式：105=103+1+1
print(band_label_loc.shape)  # (207400,105)

# 前103列的通道信息进行标准化
data_content = preprocessing.StandardScaler().fit_transform(band_label_loc[:, :-2])
# 第104列的label信息
data_label = band_label_loc[:, -2]
# 第105列的位置信息
data_loc = band_label_loc[:, -1]

# 整理最终数据矩阵
new_data = np.column_stack((data_content, data_label))
new_data = np.column_stack((new_data, data_loc))
new_data = pd.DataFrame(new_data)
new_data.to_csv('../dataset/PaviaU_band_label_loc.csv', header=False, index=False)

'''
    PaviaU_gt_band_label.csv：有label的像素整理，前103列为通道信息，104列为label
'''
# 统计label矩阵
label = np.zeros([610, 340])
for i in range(610):
    for j in range(340):
        if output_image[i][j] != 0:
            label[i][j] = output_image[i][j]

# 这次只统计有标记的像素
band_label = []
for i in range(610):
    for j in range(340):
        # 在此判断当前像素是否有标记，其余操作不变
        if label[i][j] != 0:
            currentband = list(input_image[i][j])
            currentband.append(label[i][j])
            band_label.append(currentband)

# 统计出一共42776条数据有效
band_label = np.array(band_label)
print(band_label.shape)  # (42776,104)

data_content = preprocessing.StandardScaler().fit_transform(band_label[:, :-1])
data_label = band_label[:, -1]

new_data = np.column_stack((data_content, data_label))
new_data = pd.DataFrame(new_data)

new_data.to_csv('../dataset/PaviaU_gt_band_label.csv', header=False, index=False)

'''
    PaviaU_gt_band_label_loc.csv：有label的像素整理，前103列为通道信息，104列为label，105列为像素位置
'''
# 统计label矩阵
label = np.zeros([610, 340])
for i in range(610):
    for j in range(340):
        if output_image[i][j] != 0:
            label[i][j] = output_image[i][j]

# 这次只统计有标记的像素
band_label_loc = []
for i in range(610):
    for j in range(340):
        # 在此判断当前像素是否有标记，其余操作不变
        if label[i][j] != 0:
            currentband = list(input_image[i][j])
            currentband.append(label[i][j])
            currentband.append(i * 340 + j)
            band_label_loc.append(currentband)

# 统计出一共42776条数据有效
band_label_loc = np.array(band_label_loc)
print(band_label_loc.shape)  # (42776,105)

data_content = preprocessing.StandardScaler().fit_transform(band_label_loc[:, :-2])
data_label = band_label_loc[:, -2]
data_loc = band_label_loc[:, -1]

new_data = np.column_stack((data_content, data_label))
new_data = np.column_stack((new_data, data_loc))
new_data = pd.DataFrame(new_data)

new_data.to_csv('../dataset/PaviaU_gt_band_label_loc.csv', header=False, index=False)

'''
    PaviaU_gt_band_label_loc_.csv：与PaviaU_gt_band_label_loc.csv不同，此数据的标准化是在全部数据标准化后抽取出来的
'''
# 导入gt数据集
pavia_gt = pd.read_csv('../dataset/PaviaU_gt_band_label_loc.csv', header=None)
pavia_gt = pavia_gt.as_matrix()
# 获取通道矩阵
pavia_gt_content = pavia_gt[:, :-2]
# 获取标记矩阵
pavia_gt_label = pavia_gt[:, -2]
# 获取位置矩阵
pavia_gt_loc = pavia_gt[:, -1]
print(pavia_gt_loc)

# 导入全部数据集
pavia = pd.read_csv('../dataset/PaviaU_band_label_loc.csv', header=None)
pavia = pavia.as_matrix()
# 获取通道矩阵
pavia_content = pavia[:, :-2]
# 获取标记矩阵
pavia_label = pavia[:, -2]
# 获取位置矩阵
pavia_loc = pavia[:, -1]

i = 0
data = list()

# 遍历标准化做好的数据矩阵
for row in pavia:
    # 获取当前像素位置
    row_loc = int(row[-1])
    # 如果当前像素位置存在于有效像素列表中
    if (row_loc in pavia_gt_loc):
        # 就将标准化好的数据存入
        data.append(row)
        i = i + 1

new_data = pd.DataFrame(data)
print(new_data.shape)
# (42776, 105)
new_data.to_csv('../dataset/PaviaU_gt_band_label_loc_.csv', header=False, index=False)

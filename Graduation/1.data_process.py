import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import preprocessing

# 读取原始图片paviaU
input_image = loadmat('../dataset/origin/PaviaU.mat')['paviaU']
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
band = pd.DataFrame(band)
# 生成(207400, 103)像素通道信息矩阵
band.to_csv('../dataset/PaviaU_band.csv', header=False, index=False)

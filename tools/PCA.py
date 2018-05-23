from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
print(X)

# n_components：保留的特征数，默认为1如果设置成mle，会自动确定保留的特征数
pca = PCA(n_components=1)
new_X = pca.fit_transform(X)
# print(new_X)

# 如果设置成mle，会自动确定保留的特征数
pca = PCA(n_components='mle')
new_X = pca.fit_transform(X)
# print(new_X)

# 12条数据分别分布在（1,1）,（2,2）,（3,3）,（4,4）四个点周围，可以看做4类
data = np.array([[1., 1.],
                 [0.9, 0.95],
                 [1.01, 1.03],
                 [2., 2.],
                 [2.03, 2.06],
                 [1.98, 1.89],
                 [3., 3.],
                 [3.03, 3.05],
                 [2.89, 3.1],
                 [4., 4.],
                 [4.06, 4.02],
                 [3.97, 4.01]])

pca = PCA(n_components=1)
new_data = pca.fit_transform(data)
print(new_data)


def pca_(dataMat, n):
    # 求数据矩阵每一列的均值
    meanVals = np.mean(dataMat, axis=0)
    # 数据矩阵每一列特征减去该列的特征均值
    meanRemoved = dataMat - meanVals
    # 计算协方差矩阵，除数n-1是为了得到协方差的无偏估计
    covMat = np.cov(meanRemoved.transpose())
    # 计算协方差矩阵的特征值及对应的特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # sort():对特征值矩阵排序(由小到大)
    # argsort():对特征值矩阵进行由小到大排序，返回对应排序后的索引
    eigValInd = np.argsort(eigVals)
    # 从排序后的矩阵最后一个开始自下而上选取最大的N个特征值，返回其对应的索引
    eigValInd = eigValInd[::-1]
    eigValInd = eigValInd[:n]
    # 将特征值最大的N个特征值对应索引的特征向量提取出来，组成压缩矩阵
    redEigVects = eigVects[:, eigValInd]
    # 将去除均值后的数据矩阵*压缩矩阵，转换到新的空间，使维度降低为N
    lowDDataMat = meanRemoved * redEigVects
    # 返回压缩后的数据矩阵即该矩阵反构出原始数据矩阵
    return lowDDataMat


# print(pca(X, 1))
# print(pca(data, 1))

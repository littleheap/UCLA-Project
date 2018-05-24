from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

'''
    PaviaU
'''
data = pd.read_csv('../dataset/PaviaU_gt_band.csv', header=None)
data = data.as_matrix()

# 获取特征矩阵
data_content = data[:, :-1]

pca = PCA(n_components=80)

new_content = pca.fit_transform(data_content)

print(data_content.shape)

print(new_content.shape)

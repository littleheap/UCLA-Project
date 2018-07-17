from sklearn.decomposition import PCA
import pandas as pd

'''
    PCA降维PaviaU有标记的43776条数据
'''
data = pd.read_csv('../dataset/PaviaU_gt_band.csv', header=None)

data = data.as_matrix()

data_content = data[:, :-1]

# print(data)

# 99%降维
pca = PCA(n_components=0.99)

new_content = pca.fit_transform(data_content)

print(data_content.shape)  # (42776, 103)

print(new_content.shape)  # (42776, 4)

print(new_content)

# 99.99%降维
pca = PCA(n_components=0.9999)

new_content = pca.fit_transform(data_content)

print(data_content.shape)  # (42776, 103)

print(new_content.shape)  # (42776, 49)

print(new_content)

# 自动降维
pca = PCA(n_components='mle', svd_solver='full')

new_content = pca.fit_transform(data_content)

print(data_content.shape)  # (42776, 103)

print(new_content.shape)  # (42776, 102)

print(new_content)

# 自主降维
pca = PCA(n_components=80)

new_content = pca.fit_transform(data_content)

print(data_content.shape)  # (42776, 103)

print(new_content.shape)  # (42776, 80)

print(new_content)

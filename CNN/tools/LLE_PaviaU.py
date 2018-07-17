from sklearn import manifold
import pandas as pd

'''
    LLE降维PaviaU有标记的43776条数据
'''
data = pd.read_csv('../dataset/PaviaU_gt_band.csv', header=None)

data = data.as_matrix()

data_content = data[:, :-1]

print(data_content.shape)  # (42776, 103)

# 用标准LLE做降维，近邻数设为n_neighbors，降维后的维度设置为n_components，method还可换hessian
train_data = data_content
# print(train_data.shape)  # (42776, 103)
trans_data = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=50, method='standard').fit_transform(
    train_data)
print(trans_data.shape)

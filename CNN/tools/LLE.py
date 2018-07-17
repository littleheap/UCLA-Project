import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets
from sklearn.utils import check_random_state

'''
    通过造数据测试LLE数据降维
'''

'''
    生成随机数据：LLE必须要基于流形不能闭合，因此生成了缺口三维球体
'''
# 随机变量数目
n_samples = 500
# 随机种子
random_state = check_random_state(0)
p = random_state.rand(n_samples) * (2 * np.pi - 0.55)
# print(len(p))
# print(p)
t = random_state.rand(n_samples) * np.pi
# print(len(t))
# print(t)

# 判定500个生成数据t是否符合要求，屏蔽掉不符合要求的t
indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8))))
# print(len(indices))
# p与t对应，将对应的p筛选出来
colors = p[indices]
# print(len(colors))  # 363
# 生成3维数据x,y,z
x, y, z = np.sin(t[indices]) * np.cos(p[indices]), np.sin(t[indices]) * np.sin(p[indices]), np.cos(t[indices])
# print(len(x))  # 363
# print(len(y))  # 363
# print(len(z))  # 363

# 显示原始数据
fig = plt.figure()
# 建模3D图像
ax = Axes3D(fig, elev=30, azim=-20)
ax.scatter(x, y, z, c=p[indices], marker='o', cmap=plt.cm.rainbow)
fig.show()

# 用LLE将其从3维降为2维并可视化，近邻数设为30，用标准的LLE算法
train_data = np.array([x, y, z]).T
# print(train_data.shape)  # (363, 3)
trans_data = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=2, method='standard').fit_transform(
    train_data)
# print(trans_data.shape)  # (363, 2)
plt.scatter(trans_data[:, 0], trans_data[:, 1], marker='o', c=colors)
plt.show()

# 不同的近邻数时，LLE算法降维的效果
for index, k in enumerate((10, 20, 30, 40)):
    plt.subplot(2, 2, index + 1)
    trans_data = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=2,
                                                 method='standard').fit_transform(train_data)
    plt.scatter(trans_data[:, 0], trans_data[:, 1], marker='o', c=colors)
    plt.text(.99, .01, ('LLE: k=%d' % (k)),
             transform=plt.gca().transAxes, size=10,
             horizontalalignment='right')
plt.show()

# 保持相同k近邻数，用HLLE的效果
for index, k in enumerate((10, 20, 30, 40)):
    plt.subplot(2, 2, index + 1)
    trans_data = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=2,
                                                 method='hessian').fit_transform(train_data)
    plt.scatter(trans_data[:, 0], trans_data[:, 1], marker='o', c=colors)
    plt.text(.99, .01, ('HLLE: k=%d' % (k)),
             transform=plt.gca().transAxes, size=10,
             horizontalalignment='right')
plt.show()

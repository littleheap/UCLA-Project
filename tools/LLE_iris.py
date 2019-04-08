import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold

'''
    局部线性嵌入（LLE）:保持邻域内样本之间的线性关系
　　
    输入：样本集D，近邻参数k，低维空间维数n'
    输出：样本集在低维空间中的矩阵Z
    
    算法步骤：
    1）对于样本集中的每个点x，确定其k近邻，获得其近邻下标集合Q，然后依据公式计算Wi,j
    2）根据Wi,j构建矩阵W
    3）依据公式计算M
    4）对M进行特征值分解，取其最小的n'个特征值对应的特征向量，即得到样本集在低维空间中的矩阵Z
'''


def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target


def test_LocallyLinearEmbedding(*data):
    X, Y = data
    for n in [4, 3, 2, 1]:
        lle = manifold.LocallyLinearEmbedding(n_components=n)
        lle.fit(X)
        print("reconstruction_error_(n_components=%d):%s" % (n, lle.reconstruction_error_))


def plot_LocallyLinearEmbedding_k(*data):
    X, Y = data
    Ks = [1, 5, 25, Y.size - 1]
    fig = plt.figure()
    #  colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),(0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)
    for i, k in enumerate(Ks):
        lle = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors=k)
        X_r = lle.fit_transform(X)
        ax = fig.add_subplot(2, 2, i + 1)
        colors = (
            (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.4, 0.6, 0), (0.6, 0.4, 0),
            (0, 0.6, 0.4), (0.5, 0.3, 0.2),)
        for label, color in zip(np.unique(Y), colors):
            position = Y == label
            ax.scatter(X_r[position, 0], X_r[position, 1], label="target=%d" % label, color=color)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[0]")
    ax.legend(loc="best")
    ax.set_title("k=%d" % k)
    plt.suptitle("LocallyLinearEmbedding")
    plt.show()


X, Y = load_data()
test_LocallyLinearEmbedding(X, Y)
plot_LocallyLinearEmbedding_k(X, Y)

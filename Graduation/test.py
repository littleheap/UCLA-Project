from sklearn import preprocessing
import numpy as np

# preprocessing.StandardScaler().fit_transform(X[:, :])操作示意
X = np.array([[1., -1., 2.],
              [2., 0., 0.],
              [0., 1., -1.]])
X_scaled = preprocessing.StandardScaler().fit_transform(X[:, :])
print(X_scaled)
'''
    [[ 0.         -1.22474487  1.33630621]
     [ 1.22474487  0.         -0.26726124]
     [-1.22474487  1.22474487 -1.06904497]]
'''
print(X_scaled.mean(axis=0))  # [0. 0. 0.]
print(X_scaled.std(axis=0))  # [1. 1. 1.]

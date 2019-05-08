from sklearn import preprocessing
import numpy as np
import pandas as pd

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

print([5])

li = list()

li.extend([5])
li.extend([6])

print(li[0])

print('================')

mark = []

for i in range(100):
    mark.append(i)

print(mark)

mark.remove(0)
mark.remove(1)

print(mark)

print(0 in mark)

print(5 in mark)

print('===============')

a = set()
a.add(1)
a.add(2)
a.add(1)
print(a)
print(len(a))

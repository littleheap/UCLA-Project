import numpy as np
import pandas as pd

# print(np.array([[1, 2, 3], [4, 5, 6]]))

# print(np.column_stack([np.array([[1, 2, 3], [4, 5, 6]]), np.array([0, 0])]))

# print(pd.DataFrame(np.column_stack([np.array([[1, 2, 3], [4, 5, 6]]), np.array([0, 0])])))

# 导入数据集切割训练与测试数据
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

# print(371775//610)

# print(1==1==1.0==1)

# input_image = loadmat('./dataset/PaviaU.mat')['paviaU']

# print(input_image[2, 3, :])
# print(input_image[2, 3, :].shape)


# a = np.array([1, 2, 3, 4, 5])
#
# print(a)
#
# print(a.reshape(-1, 1))

dic = {1: 2, 2: 3, 3: 4}
print(dic[1])

from sklearn import *
from sklearn.svm import SVC
from sklearn.model_selection import *
import time
import joblib
import numpy as np
import pandas as pd

'''
    训练并存储模型，观察分类效果，在图中显示与原图对比
'''

# 导入数据集切割训练与测试数据，用最终版本
data = pd.read_csv('../dataset/PaviaU_gt_band_label_loc_.csv', header=None)
data = data.as_matrix()

# 获取特征矩阵
data_band = data[:, :-2]
# 获取标记矩阵
data_label = data[:, -2]

# test测试集
data = pd.read_csv('../dataset/CNN_test_40976_shuffle.csv', header=None)
data = data.as_matrix()

# 获取test特征矩阵
data_test = data[:, :-1]
print(data_test.shape)  # (40976, 103)

# 获取test的标记
label_test = data[:, -1]
print(label_test.shape)  # (40976,)

# train训练集
data = pd.read_csv('../dataset/CNN_train_1800_shuffle.csv', header=None)
data = data.as_matrix()

# 获取train特征矩阵
data_train = data[:, :-1]
print(data_train.shape)  # (1800, 103)

# 获取train标记矩阵
label_train = data[:, -1]
print(label_train.shape)  # (1800,)

print(data_train.shape)  # (1800, 103)
print(data_test.shape)  # (40976, 103)
print(label_train.shape)  # (1800,)
print(label_test.shape)  # (40976,)


# 计算正确率函数
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc = ('%.4f' % np.mean(acc))
    print(tip + '正确率：', (float(acc) * 100), '%')


# 返回错误坐标
def return_wrong(a, b):
    acc = a.ravel() == b.ravel()
    wrong = list()
    i = 0
    for target in acc:
        if target:
            i = i + 1
        else:
            i = i + 1
            wrong.append(i)
    return wrong


'''
    (1)SVC支撑向量分类器
'''
time_start = time.time()

# 初始化支撑向量分类器
model = SVC(kernel='rbf', gamma=0.02783, C=100)
# 训练模型
model.fit(data_train, label_train)
# 预测测试集
pred = model.predict(data_band)

# 计算精确度
accuracy = metrics.accuracy_score(data_label, pred) * 100
print(accuracy, '%')  # 90.43098399062866 %

# 存储模型
joblib.dump(model, './models/SVC_1800.m')

time_end = time.time()
print('totally cost', time_end - time_start)  # 4s

'''
    (2)SVC+CV
'''
# time_start = time.time()
#
# # 初始化支撑向量分类器
# model = SVC(kernel='rbf')
#
# # c取0.01~100
# c_can = np.logspace(-2, 2, 10)
# # c_can = [16]
#
# # gamma取0.01~100
# gamma_can = np.logspace(-2, 2, 10)
# # gamma_can=[0.125]
#
# # 交叉验证
# model = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=2)
#
# # 训练模型
# model.fit(data_train, label_train)
#
# # 打印最优超参数
# print('验证参数：\n', model.best_params_)
# # >>>  {'C': 100.0, 'gamma': 0.027825594022071243}
#
# # 预测测试集
# pred = model.predict(data_test)
#
# # 计算精确度
# accuracy = metrics.accuracy_score(label_test, pred) * 100
# show_accuracy(pred, label_test, 'SVC_CV')
#
# # 计算全集
# show_accuracy(pred, label_test, 'SVC_CV')
#
# # 查找错误分类坐标
# wrong = return_wrong(pred, label_test)
#
# # print(wrong)
# print(len(wrong))  # 3923
#
# # 存储模型
# joblib.dump(model, './models/SVC_CV.m')
#
# time_end = time.time()
# print('totally cost', time_end - time_start)  # 63.3s

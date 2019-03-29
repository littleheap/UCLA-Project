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

# 导入数据集切割训练与测试数据
data = pd.read_csv('../dataset/PaviaU_gt_band_label.csv', header=None)
data = data.as_matrix()

# 获取特征矩阵
data_content = data[:, :-1]
# 获取标记矩阵
data_label = data[:, -1]

# 切割训练集测试集
data_train, data_test, label_train, label_test = train_test_split(data_content, data_label, test_size=0.3)

print(data_train.shape)  # (29943, 103)
print(data_test.shape)  # (12833, 103)
print(label_train.shape)  # (29943,)
print(label_test.shape)  # (12833,)


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
pred = model.predict(data_test)

# 计算精确度
accuracy = metrics.accuracy_score(label_test, pred) * 100
print(accuracy, '%')  # 96.29%

# 存储模型
joblib.dump(model, './models/SVC.m')

time_end = time.time()
print('totally cost', time_end - time_start)  # 23s

'''
    (2)SVC+CV
'''
time_start = time.time()

# 初始化支撑向量分类器
model = SVC(kernel='rbf')

# c取0.01~100
c_can = np.logspace(-2, 2, 10)
# c_can = [16]

# gamma取0.01~100
gamma_can = np.logspace(-2, 2, 10)
# gamma_can=[0.125]

# 交叉验证
model = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=2)

# 训练模型
model.fit(data_train, label_train)

# 打印最优超参数
print('验证参数：\n', model.best_params_)
# >>>  {'C': 100.0, 'gamma': 0.027825594022071243}

# 预测测试集
pred = model.predict(data_test)
# 预测全集
pred = model.predict(data_content)

# 计算精确度
accuracy = metrics.accuracy_score(label_test, pred) * 100
show_accuracy(pred, label_test, 'SVC_CV')

# 计算全集
show_accuracy(pred, data_label, 'SVC_CV')

# 查找错误分类坐标
wrong = return_wrong(pred, data_label, 'SVC_CV')

print(wrong)
print(len(wrong))

# 存储模型
joblib.dump(model, './models/SVC_CV.m')

time_end = time.time()
print('totally cost', time_end - time_start)  # 4343s

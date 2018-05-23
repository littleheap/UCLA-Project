from functools import reduce
import matplotlib.pyplot as plt
import time
from scipy.io import *
import spectral
from spectral import *
import joblib
from sklearn.model_selection import *
import numpy as np
from sklearn.svm import SVC
from sklearn import *
import pandas as pd

'''
    数据集PaviaU做分类程序：
    数据集：PaviaU（610*340*103）
    标记集：PaviaU_gt（610*340）
    步骤：
    1.加载遥感图像的.mat数据进行呈图显示
    2.将.mat的数据转化为python后续算法处理的csv文件
    3.训练并存储模型，观察分类效果，在图中显示与原图对比
'''

''' 
    #################################
    1.加载遥感图像的.mat数据进行呈图显示
    #################################
'''
input_image = loadmat('./dataset/PaviaU.mat')['paviaU']

output_image = loadmat('./dataset/PaviaU_gt.mat')['paviaU_gt']

print(input_image.shape)  # (610, 340, 103)

print(output_image.shape)  # (610, 340)

print(np.unique(input_image))  # [   0    1    2 ... 7996 7997 8000]

print(np.unique(output_image))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

'''
    统计类元素个数
'''
dict_category = {}

for i in range(output_image.shape[0]):  # 610
    for j in range(output_image.shape[1]):  # 340
        if output_image[i][j] in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            if output_image[i][j] not in dict_category:
                dict_category[output_image[i][j]] = 0
            dict_category[output_image[i][j]] += 1

# {0: 164624, 1: 6631, 2: 18649, 3: 2099, 4: 3064, 5: 1345, 6: 5029, 7: 1330, 8: 3682, 9: 947}
print(dict_category)

# 验证全部类别恰好为207400 (610 * 340 = 164624 + 42776)
print(reduce(lambda x, y: x + y, dict_category.values()))

'''
    光谱图像展示，见PaviaU_gt.py和paviaU_gt.png
'''
ground_truth = imshow(classes=output_image.astype(int), figsize=(9, 9))

ksc_color = np.array([[255, 255, 255],
                      [184, 40, 99],
                      [74, 77, 145],
                      [35, 102, 193],
                      [238, 110, 105],
                      [117, 249, 76],
                      [114, 251, 253],
                      [126, 196, 59],
                      [234, 65, 247],
                      [141, 79, 77],
                      [183, 40, 99],
                      [0, 39, 245],
                      [90, 196, 111],
                      ])

ground_truth = imshow(classes=output_image.astype(int), figsize=(9, 9), colors=ksc_color)

'''
    ############################################
    2.将.mat的数据转化为python后续算法处理的csv文件
    ############################################
'''

# 提取分类类别和对应1*1*103通道矩阵，整合通道+类别
# 初始化label矩阵
label = np.zeros([610, 340])

# 提取label矩阵
for i in range(610):
    for j in range(340):
        if output_image[i][j] != 0:
            label[i][j] = output_image[i][j]

# 初始化通道类别列表
bandwithlabel = []

# 提取有类别标记的通道，并统计计入bandwithlabel
for i in range(610):
    for j in range(340):
        if label[i][j] != 0:
            currentband = list(input_image[i][j])
            currentband.append(label[i][j])
            bandwithlabel.append(currentband)

# 转化为矩阵
bandwithlabel = np.array(bandwithlabel)

# 数据维度+通道和标签维度，数据42776维度（1-9类别），通道103个波段，最后104列是标签
print(bandwithlabel.shape)  # (42776,104)

'''
    数据标准化处理
'''
# 标准化预处理特征数据，不考虑最后一列标签（42776,103）
data_content = preprocessing.StandardScaler().fit_transform(bandwithlabel[:, :-1])
# data_D = preprocessing.MinMaxScaler().fit_transform(new_datawithlabel_array[:,:-1])

# 将最后一列标签单独抽取（42776,1）
data_label = bandwithlabel[:, -1]

# 合并标准化特征矩阵和标记矩阵
new_data = np.column_stack((data_content, data_label))
# 将全新标准化矩阵转化为CSV行列阵格式
new_data = pd.DataFrame(new_data)
# 存储行列阵
# new_data.to_csv('./dataset/PaviaU.csv', header=False, index=False)

'''
    ###############################################
    3.训练并存储模型，观察分类效果，在图中显示与原图对比
    ###############################################
'''

'''
    训练模型并存储模型
'''
# 导入数据集切割训练与测试数据
data = pd.read_csv('./dataset/PaviaU.csv', header=None)
data = data.as_matrix()

# 获取特征矩阵
data_content = data[:, :-1]

# 获取标记矩阵
data_label = data[:, -1]

# 切割训练集测试集
data_train, data_test, label_train, label_test = train_test_split(data_content, data_label, test_size=0.3)


# print(data_train.shape)  # (29943, 103)
# print(data_test.shape)  # (12833, 103)
# print(label_train.shape)  # (29943,)
# print(label_test.shape)  # (12833,)


# 模型训练与拟合

# 计算正确率
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc = ('%.4f' % np.mean(acc))
    print(tip + '正确率：', (float(acc) * 100), '%')


# 返回错误坐标
def return_wrong(a, b, tip):
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
# time_start = time.time()
#
# # 初始化支撑向量分类器
# model = SVC(kernel='rbf', gamma=0.02783, C=100)
# # 训练模型
# model.fit(data_train, label_train)
# # 预测测试集
# pred = model.predict(data_test)
#
# # 计算精确度
# accuracy = metrics.accuracy_score(label_test, pred) * 100
# print(accuracy, '%')  # 96.29%
#
# # 存储模型
# joblib.dump(model, './models/PaviaU/SVC.m')
#
# time_end = time.time()
# print('totally cost', time_end - time_start)  # 20s

'''
    (2)SVC+CV
'''
time_start = time.time()

# # 初始化支撑向量分类器
# model = SVC(kernel='rbf')
#
# # c 取 0.01~100
# c_can = np.logspace(-2, 2, 10)
# # c_can = [16]
#
# # gamma 取 0.01~100
# gamma_can = np.logspace(-2, 2, 10)
# # gamma_can = [0.125]
#
# # 交叉验证
# model = GridSearchCV(model, param_grid={'C': c_can, 'gamma': gamma_can}, cv=2)
#
# # 训练模型
# model.fit(data_train, label_train)

# # 打印最优超参数
# print('验证参数：\n', model.best_params_)
# # >>>  {'C': 100.0, 'gamma': 0.027825594022071243}

# 加载模型
model = joblib.load('./models/PaviaU/SVC_CV.m')

# 预测测试集
# pred = model.predict(data_test)
# 预测全集
pred = model.predict(data_content)

# 计算精确度
# accuracy = metrics.accuracy_score(label_test, pred) * 100
# show_accuracy(pred, label_test, 'SVC_CV')
# 计算全集
show_accuracy(pred, data_label, 'SVC_CV')

# 查找错误分类坐标
wrong = return_wrong(pred, data_label, 'SVC_CV')

print(wrong)
print(len(wrong))

# 存储模型
# joblib.dump(model, './models/PaviaU/SVC_CV.m')

time_end = time.time()
print('totally cost', time_end - time_start)  # 4343s

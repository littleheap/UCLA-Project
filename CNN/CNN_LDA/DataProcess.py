import random
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

'''
    1.将原有标记的42776条数据归类，从1-9，通道全部降维到49，并将每个类别内部数据shuffle乱序，存储到一个csv中
    ---./dataset/reduce_data.csv
'''

# # 导入数据集（用SVM中最后处理出的最标准的数据，标准化统一）
# data = pd.read_csv('../../dataset/PaviaU_gt_band_label_loc_.csv', header=None)
# data = data.as_matrix()
#
# # 获取特征矩阵
# data_content = data[:, :-2]
# # print(data_content.shape)  # (42776, 103)
#
# # 获取标记矩阵
# data_label = data[:, -2]
# # print(data_label.shape)  # (42776,)
#
# # LDA降维到8维度
# lda = LinearDiscriminantAnalysis(n_components=49)
# lda.fit(data_content, data_label)
# new_data_content = lda.transform(data_content)
# print(new_data_content.shape)  # (42776, 8)
#
# # 所有42776条数据通道list
# allband = list()
#
# # 生成0-9一共10个类别的通道子list
# for i in range(10):
#     allband.append(list())
#
# print(allband)  # [[], [], [], [], [], [], [], [], [], []]
#
# print(len(allband))  # 10
#
# # 遍历42776条数据
# for i in range(42776):
#     # 获取当前数据label
#     currentlabel = int(data_label[i])
#     # 0类别就跳过
#     if (currentlabel == 0):
#         continue
#     # 获取当前数据通道
#     currentband = list(new_data_content[i])
#     # 将通道末尾添加对应标记
#     currentband.append(currentlabel)
#     # 将对应通道添加到对应类别list中
#     allband[currentlabel].append(currentband)
#
# # 成功获取有标记9个类别通道并归类
# # for i in range(10):
# #     print(i, ':', len(allband[i]), end=' | ')
# # 0 : 0 | 1 : 6631 | 2 : 18649 | 3 : 2099 | 4 : 3064 | 5 : 1345 | 6 : 5029 | 7 : 1330 | 8 : 3682 | 9 : 947 |
#
#
# # 初始话写入csv的数据列表
# data_csv = list()
# # 由于shuffle操作有随机性，所以先将shuffle后的数据存储到csv固定下来
# for i in range(10):
#     # 0类跳过
#     if i == 0:
#         continue
#     # 打乱当前类别数据
#     random.shuffle(allband[i])
#     # 读取当前类别数据打乱后的每一行
#     for j in range(len(allband[i])):
#         # 获取当前行数据
#         row = allband[i][j]
#         data_csv.append(row)
#
# new_data_csv = pd.DataFrame(data_csv)
#
# print(new_data_csv.shape)  # (42776, 8+1)
#
# new_data_csv.to_csv('./dataset/reduce_data.csv', header=False, index=False)

'''
    2.从上面分类后的数据集中抽取9个类别训练数据，每个类别200条，一共1800条数据，每条数据8通道信息+label，
    一共(1800,9)，以及测试数据，每个类别除去测试集200条，剩余的条数作为该类别的测试数据，一共(N,9)
    Train和Test两个数据集分别存储成csv格式，方便读取。
    ---./dataset/reduce_data_train.csv
    ---./dataset/reduce_data_train.csv
'''

# # 导入上面分类好有序的42776条LDA降维数据
# data = pd.read_csv('./dataset/reduce_data.csv', header=None)
# data = data.as_matrix()
#
# # 初始话写入csv的数据列表
# data_train_csv = list()
# data_test_csv = list()
# # 类别游标
# cursor_class = 1
# cursor_number = 0
# # 遍历全部数据
# for i in range(42776):
#     # 读取当前行
#     row = data[i]
#     # 读取当前数据label
#     current_label = row[-1]
#     # 每个类别前200条作为Train训练数据
#     if cursor_number < 200:
#         data_train_csv.append(row)
#         cursor_number = cursor_number + 1
#     # 每个类别剩余作为Test数据
#     elif current_label == cursor_class:
#         data_test_csv.append(row)
#     # 进入下一个类别后，更新游标
#     else:
#         cursor_class = cursor_class + 1
#         data_train_csv.append(row)
#         cursor_number = 1
#
# new_data_train_csv = pd.DataFrame(data_train_csv)
#
# print(new_data_train_csv.shape)  # (1800, 9)
#
# new_data_train_csv.to_csv('./dataset/reduce_data_train.csv', header=False, index=False)
#
# new_data_test_csv = pd.DataFrame(data_test_csv)
#
# print(new_data_test_csv.shape)  # (40976, 9) = 42776-1800
#
# new_data_test_csv.to_csv('./dataset/reduce_data_test.csv', header=False, index=False)

'''
    3.将train和test数据集shuffle打乱顺序
    ---./dataset/reduce_data_train_shuffle.csv
    ---./dataset/reduce_data_test_shuffle.csv
'''

# # train数据集
# train_data = pd.read_csv('./dataset/reduce_data_train.csv', header=None)
#
# train_data = train_data.as_matrix()
#
# train_list = list()
#
# for i in range(1800):
#     train_list.append(list(train_data[i]))
#
# random.shuffle(train_list)
#
# new_train_list = pd.DataFrame(train_list)
#
# print(new_train_list.shape)  # (1800, 9)
#
# new_train_list.to_csv('./dataset/reduce_data_train_shuffle.csv', header=False, index=False)
#
# # test数据集
# test_data = pd.read_csv('./dataset/reduce_data_test.csv', header=None)
#
# test_data = test_data.as_matrix()
#
# test_list = list()
#
# for i in range(40976):
#     test_list.append(list(test_data[i]))
#
# random.shuffle(test_list)
#
# new_test_list = pd.DataFrame(test_list)
#
# print(new_test_list.shape)  # (40976, 9)
#
# new_test_list.to_csv('./dataset/reduce_data_test_shuffle.csv', header=False, index=False)

'''
    4.将train和test标签转化为onehot矩阵格式，并存储
    ---./dataset/reduce_data_train_shuffle_label_onehot.csv
    ---./dataset/reduce_data_test_shuffle_label_onehot.csv
'''


# # onehot函数
# def onehot(labels, length):
#     sess = tf.Session()
#     batch_size = tf.size(labels)
#     labels = tf.expand_dims(labels, 1)
#     indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
#     concated = tf.concat([indices, labels], 1)
#     onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, length]), 1.0, 0.0)
#     return onehot_labels
#
#
# # onehot([1, 3, 5, 7, 9], 10)
#
# # train数据集
# train_data = pd.read_csv('./dataset/reduce_data_train_shuffle.csv', header=None)
# train_data = train_data.as_matrix()
#
# # 获取特征矩阵
# data_content = train_data[:, :-1]
# print(data_content.shape)  # (1800, 8)
#
# # 获取标记矩阵
# data_label = train_data[:, -1]
# print(data_label.shape)  # (1800,)
#
# train_label = []
#
# for i in range(1800):
#     train_label.append(int(data_label[i]))
#
# onehot_label = onehot(train_label, 10)
#
# print(onehot_label.shape)  # (1800, 10)
#
# onehot_csv = []
#
# sess = tf.Session()
#
# onehot_label_numpy = onehot_label.eval(session=sess)
#
# print(onehot_label_numpy)
#
# print(onehot_label_numpy.shape)  # (1800, 10)
#
# for i in range(1800):
#     onehot_csv.append(onehot_label_numpy[i])
#
# onehot_csv = pd.DataFrame(onehot_csv)
#
# onehot_csv.to_csv('./dataset/reduce_data_train_shuffle_label_onehot.csv', header=False, index=False)
#
# # test数据集
# test_data = pd.read_csv('./dataset/reduce_data_test_shuffle.csv', header=None)
# test_data = test_data.as_matrix()
#
# # 获取特征矩阵
# data_content = test_data[:, :-1]
# print(data_content.shape)  # (40976, 8)
#
# # 获取标记矩阵
# data_label = test_data[:, -1]
# print(data_label.shape)  # (40976,)
#
# test_label = []
#
# for i in range(40976):
#     test_label.append(int(data_label[i]))
#
# onehot_label = onehot(test_label, 10)
#
# print(onehot_label.shape)  # (40976, 10)
#
# onehot_csv = []
#
# sess = tf.Session()
#
# onehot_label_numpy = onehot_label.eval(session=sess)
#
# print(onehot_label_numpy)
#
# print(onehot_label_numpy.shape)  # (40976, 10)
#
# for i in range(40976):
#     onehot_csv.append(onehot_label_numpy[i])
#
# onehot_csv = pd.DataFrame(onehot_csv)
#
# onehot_csv.to_csv('./dataset/reduce_data_test_shuffle_label_onehot.csv', header=False, index=False)

'''
    5.分别按批次读取train训练集通道数据和onehot标记数据，过程操作
'''

# train训练集
data_train = pd.read_csv('./dataset/reduce_data_train_shuffle.csv', header=None)
data_train = data_train.as_matrix()

# 获取train特征矩阵
data_train_band = data_train[:, :-1]
print(data_train_band.shape)  # (1800, 8)

# 获取train标记onehot矩阵
data_train_label = pd.read_csv('./dataset/reduce_data_train_shuffle_label_onehot.csv', header=None)
data_train_label = data_train_label.as_matrix()
print(data_train_label.shape)  # (1800, 10)


# batch size后期设定100
# 获取100个1800以内的随机数
def get_random_100():
    random_100 = []
    while (len(random_100) < 100):
        x = random.randint(0, 1799)
        if x not in random_100:
            random_100.append(x)
    return random_100


random_100 = get_random_100()

# 随机获取100个band和对应的label
data_band_batch_100 = data_train_band[random_100]
data_label_batch_100 = data_train_label[random_100]

# 验证尺寸
print(data_band_batch_100.shape)  # (100, 8)
print(data_label_batch_100.shape)  # (100, 10)

'''
    6.读取test测试集通道数据和onehot标记数据，过程操作
'''

# test测试集
data_test = pd.read_csv('./dataset/reduce_data_test_shuffle.csv', header=None)
data_test = data_test.as_matrix()

# 获取test特征矩阵
data_test_band = data_test[:, :-1]
print(data_test_band.shape)  # (40976, 8)

# 获取test标记onehot矩阵
data_test_label = pd.read_csv('./dataset/reduce_data_test_shuffle_label_onehot.csv', header=None)
data_test_label = data_test_label.as_matrix()
print(data_test_label.shape)  # (40976, 10)

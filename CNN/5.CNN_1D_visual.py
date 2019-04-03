import cv2
import random
from scipy.io import *
import pandas as pd
import numpy as np
import tensorflow as tf

'''
    实现基础CNN网络
'''

# 每个批次的大小
batch_size = 100

# 计算一共有多少个训练批次遍历一次训练集
n_batch = 1800 // batch_size  # 18

# 读取全集
data_all = pd.read_csv('../dataset/PaviaU_gt_band_label_loc_.csv', header=None)
data_all = data_all.as_matrix()

# 获取全集特征矩阵
data_all_band = data_all[:, :-2]
print(data_all_band.shape)  # (42776, 103)
# 取前100个通道
data_all_band = data_all_band[:, :100]
print(data_all_band.shape)  # (42776, 100)

# 获取标记矩阵
data_all_label = data_all_band[:, -2]

# 获取位置矩阵
data_all_loc = data_all[:, -1]

# # 获取test标记onehot矩阵
# data_all_label_onehot = pd.read_csv('../dataset/CNN_label_42776_onehot.csv', header=None)
# data_all_label_onehot = data_all_label_onehot.as_matrix()
# print(data_all_label_onehot.shape)  # (42776, 10)
#
#
# # 参数概要
# def variable_summaries(var):
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean', mean)  # 平均值
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.summary.scalar('stddev', stddev)  # 标准差
#         tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
#         tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
#         tf.summary.histogram('histogram', var)  # 直方图
#
#
# # 初始化权值
# def weight_variable(shape, name):
#     initial = tf.truncated_normal(shape, stddev=0.1)  # 生成一个截断的正态分布，标准差为0.1
#     return tf.Variable(initial, name=name)
#
#
# # 初始化偏置
# def bias_variable(shape, name):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial, name=name)
#
#
# # 卷积层
# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 2, 1], padding='VALID')
#
#
# # 池化层
# def max_pool_2x2_first(x):
#     return tf.nn.max_pool(x, ksize=[1, 1, 7, 1], strides=[1, 1, 7, 1], padding='VALID')
#
#
# # 输入层
# with tf.name_scope('input'):
#     # 定义两个placeholder
#     x = tf.placeholder(tf.float32, [None, 100], name='x-input')
#     y = tf.placeholder(tf.float32, [None, 10], name='y-input')
#     with tf.name_scope('x_image'):
#         '''
#         改变x的格式转为2维的向量 [batch, in_height, in_width, in_channels]
#         (1)批次  (2)二维高  (3)二维宽  (4)通道数：黑白为1，彩色为3
#         '''
#         # 此处将处理好100维度的数据reshape重新折叠成1*100维度
#         x_image = tf.reshape(x, [-1, 1, 100, 1], name='x_image')
#
# # 第一层：卷积+激活+池化
# with tf.name_scope('Conv1'):
#     # 初始化第一层的W和b
#     with tf.name_scope('W_conv1'):
#         W_conv1 = weight_variable([1, 4, 1, 32], name='W_conv1')  # 1*4的采样窗口，32个卷积核从1个平面抽取特征
#         variable_summaries(W_conv1)
#     with tf.name_scope('b_conv1'):
#         b_conv1 = bias_variable([32], name='b_conv1')  # 每一个卷积核一个偏置值
#         variable_summaries(b_conv1)
#
#     # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
#     with tf.name_scope('conv2d_1'):
#         conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
#     with tf.name_scope('relu'):
#         h_conv1 = tf.nn.relu(conv2d_1)
#     with tf.name_scope('h_pool1'):
#         h_pool1 = max_pool_2x2_first(h_conv1)  # 进行max-pooling
#
# # 1*100的图片第一次卷积后还是1*49，第一次激活后不变，第一次池化后变为1*7
# # 进过上面操作后得到32张1*7的平面
#
# # 全连接层1阶段
# with tf.name_scope('fc1'):
#     # 初始化第一个全连接层的权值
#     with tf.name_scope('W_fc1'):
#         W_fc1 = weight_variable([1 * 7 * 32, 128], name='W_fc1')  # 输入层有1*7*32个列的属性，全连接层有128个隐藏神经元
#         variable_summaries(W_fc1)
#     with tf.name_scope('b_fc1'):
#         b_fc1 = bias_variable([128], name='b_fc1')  # 1024个节点
#         variable_summaries(b_fc1)
#
#     # 把第二层的输出扁平化为1维，-1代表任意值
#     with tf.name_scope('h_pool1_flat'):
#         h_pool2_flat = tf.reshape(h_pool1, [-1, 1 * 7 * 32], name='h_pool1_flat')
#     # 求第一个全连接层的输出
#     with tf.name_scope('wx_plus_b1'):
#         wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
#     with tf.name_scope('relu'):
#         h_fc1 = tf.nn.relu(wx_plus_b1)
#
#     # Dropout处理，keep_prob用来表示处于激活状态的神经元比例
#     with tf.name_scope('keep_prob'):
#         keep_prob = tf.placeholder(tf.float32, name='keep_prob')
#     with tf.name_scope('h_fc1_drop'):
#         h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')
#
# # 全连接层2阶段
# with tf.name_scope('fc2'):
#     # 初始化第二个全连接层
#     with tf.name_scope('W_fc2'):
#         W_fc2 = weight_variable([128, 10], name='W_fc2')  # 输入为128个隐藏层神经元，输出层为10个数字可能结果
#         variable_summaries(W_fc2)
#     with tf.name_scope('b_fc2'):
#         b_fc2 = bias_variable([10], name='b_fc2')
#         variable_summaries(b_fc2)
#     with tf.name_scope('wx_plus_b2'):
#         wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
#     with tf.name_scope('softmax'):
#         # 计算输出
#         prediction = tf.nn.softmax(wx_plus_b2)
#
# # 求准确率
# with tf.name_scope('accuracy'):
#     with tf.name_scope('correct_prediction'):
#         # 结果存放在一个布尔列表中
#         correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # argmax返回一维张量中最大的值所在的位置
#     with tf.name_scope('accuracy'):
#         # 求准确率
#         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#         tf.summary.scalar('accuracy', accuracy)
#
#
# # 返回错误坐标
# def return_wrong(a, b):
#     acc = a.ravel() == b.ravel()
#     # print(acc.shape)  # (42776,)
#     wrong_set = set()
#     i = 0
#     for cur in acc:
#         if cur:
#             pass
#         else:
#             cur_loc = data_all_loc[i]
#             wrong_set.add(int(cur_loc))
#         i = i + 1
#     return wrong_set
#
#
# # 返回一个像素坐标的上下左右四个坐标
# def return_4loc(loc):
#     # 获取当前预测错误的像素坐标
#     i = int(loc // 340)
#     j = int(loc % 340)
#     # up
#     i_up = i - 1
#     j_up = j
#     # down
#     i_down = i + 1
#     j_down = j
#     # left
#     i_left = i
#     j_left = j - 1
#     # right
#     i_right = i
#     j_right = j + 1
#     if i_up < 0 or i_down > 609 or j_left < 0 or j_right > 339:
#         return False, False, False, False
#     else:
#         return [i_up, j_up], [i_down, j_down], [i_left, j_left], [i_right, j_right]
#
#
# # 初始化变量
# init = tf.global_variables_initializer()
#
# # 训练模型提取
# saver = tf.train.Saver()
#
# # with tf.Session() as sess:
# #     sess.run(init)
# #     saver.restore(sess, './models/1000/CNN.ckpt')
# #     test_acc = sess.run(accuracy, feed_dict={x: data_all_band, y: data_all_label_onehot, keep_prob: 1.0})
# #     print(" Accuracy = " + str(test_acc))  # Accuracy = 0.83373857
#
# with tf.Session() as sess:
#     sess.run(init)
#     saver.restore(sess, './models/1000/CNN.ckpt')
#     right = 0
#     # 每个像素判定正误标记矩阵
#     pred = list()
#     # 每个像素的判定结果
#     pred_label = list()
#     # 逐一遍历进行分类
#     for i in range(42776):
#         # 提取单独数据后需要增维
#         band = data_all_band[i][np.newaxis, :]
#         label = data_all_label_onehot[i][np.newaxis, :]
#         # 获取当前像素判定是否正确
#         acc = sess.run(accuracy, feed_dict={x: band, y: label, keep_prob: 1.0})
#         # 将当前判定的像素分类结果统计起来（时间开销大）
#         # pred_label.extend(sess.run(tf.argmax(prediction, 1), feed_dict={x: band, y: label, keep_prob: 1.0}))
#         if acc == 1:
#             right = right + 1
#             # 如果判定正确则标记当前为1
#             pred.append(1)
#         else:
#             # 判定错误的标记为0
#             pred.append(0)
#         print(i)
#
#     print(right / 42776)  # 0.8337385449784926 此处的正确率验证上述操作没有问题
#     print(len(pred))
#     # print(len(pred_label))
#
#     # pred_label = pd.DataFrame(pred_label)
#     # pred_label.to_csv('../dataset/pred_label.csv', header=False, index=False)
#     #
#     # pred = pd.DataFrame(pred)
#     # pred.to_csv('../dataset/pred.csv', header=False, index=False)

# 获取判定结果正误统计
pred = pd.read_csv('../dataset/pred.csv', header=None)
pred = pred.as_matrix()
# 转置处理
pred = np.transpose(pred)

print(pred[0].shape)  # (42776,)
pred = pred[0]


# 计算正确率
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    acc = ('%.4f' % np.mean(acc))
    print(tip + '正确率：', (float(acc) * 100), '%')


# 返回错误坐标
def return_wrong(a, b):
    acc = a.ravel() == b.ravel()
    wrong = list()
    i = 0
    for cur in acc:
        if cur:
            pass
        else:
            cur_loc = data_all_loc[i]
            wrong.append(cur_loc)
        i = i + 1
    return wrong


# 构造辅助矩阵用于统计正误位置
temp_list = [1] * 42776
temp_list = np.array(temp_list)

# 获取错误分类坐标
wrong = return_wrong(pred, temp_list)

print(len(wrong))  # 7112 此处验证没有问题

''' 
    单颜色画图并将错误的分类像素标记CNN_WrongClass.png
'''

# 读取标记图片paviaU_gt
output_image = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']  # (610, 340)

# # 单颜色（黄色）显示标记图片
#
# # 初始化个通道，用于生成新的paviaU_gt
# c1 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']
# c2 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']
# c3 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']
#
# # 现将全部分类坐标用黄色标记
# for i in range(610):
#     for j in range(340):
#         if (output_image[i][j] == 0):
#             c1[i][j] = 255
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 1):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 2):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 3):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 4):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 5):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 6):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 7):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 8):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#         if (output_image[i][j] == 9):
#             c1[i][j] = 0
#             c2[i][j] = 255
#             c3[i][j] = 255
#
# # 将错误分类坐标用红色标记
# for value in wrong:
#     i = int(value // 340)
#     j = int(value % 340)
#     c1[i][j] = 255
#     c2[i][j] = 0
#     c3[i][j] = 255
#
# # 合并三个通道，组成三通道RGB图片
# single_merged = cv2.merge([c1, c2, c3])
# # 存储图片
# cv2.imwrite('../imgs/CNN_WrongClass.png', single_merged)
# # 显示图片
# cv2.imshow("output", single_merged)
# # 不闪退
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
    多颜色画图并将错误的分类像素标记CNN_WrongClass_multicolor.png
'''
# 导入gt数据集
pavia_gt = pd.read_csv('../dataset/PaviaU_gt_band_label_loc.csv', header=None)
pavia_gt = pavia_gt.as_matrix()

# 获取特征矩阵
pavia_gt_content = pavia_gt[:, :-2]
# 获取标记矩阵
pavia_gt_label = pavia_gt[:, -2]
# 获取位置矩阵
pavia_gt_loc = pavia_gt[:, -1]

# 导入全部数据集
pavia = pd.read_csv('../dataset/PaviaU_band_label_loc.csv', header=None)
pavia = pavia.as_matrix()

# 获取通道矩阵
pavia_content = pavia[:, :-2]
# 获取标记矩阵
pavia_label = pavia[:, -2]
# 获取位置矩阵
pavia_loc = pavia[:, -1]

# 初始化个通道，用于生成新的paviaU_gt
c1 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']
c2 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']
c3 = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']

cursor = 0

# 现将全部分类坐标用9种彩色标记，0背景类别用白色
for i in range(610):
    for j in range(340):
        if (output_image[i][j] == 0):
            c1[i][j] = 255
            c2[i][j] = 255
            c3[i][j] = 255
            continue
        if (output_image[i][j] == 1):
            c1[i][j] = 20
            c2[i][j] = 104
            c3[i][j] = 82
        if (output_image[i][j] == 2):
            c1[i][j] = 40
            c2[i][j] = 200
            c3[i][j] = 160
        if (output_image[i][j] == 3):
            c1[i][j] = 60
            c2[i][j] = 240
            c3[i][j] = 111
        if (output_image[i][j] == 4):
            c1[i][j] = 80
            c2[i][j] = 77
            c3[i][j] = 190
        if (output_image[i][j] == 5):
            c1[i][j] = 14
            c2[i][j] = 80
            c3[i][j] = 90
        if (output_image[i][j] == 6):
            c1[i][j] = 120
            c2[i][j] = 60
            c3[i][j] = 150
        if (output_image[i][j] == 7):
            c1[i][j] = 140
            c2[i][j] = 200
            c3[i][j] = 255
        if (output_image[i][j] == 8):
            c1[i][j] = 160
            c2[i][j] = 5
            c3[i][j] = 100
        if (output_image[i][j] == 9):
            c1[i][j] = 180
            c2[i][j] = 180
            c3[i][j] = 255
        cursor = cursor + 1

# 将错误分类坐标用黑色标记
for value in wrong:
    j = int(value % 340)
    i = int(value // 340)
    c1[i][j] = 0
    c2[i][j] = 0
    c3[i][j] = 0

# 合并三个通道，组成三通道RGB图片
single_merged = cv2.merge([c1, c2, c3])
# 存储图片
cv2.imwrite('../imgs/CNN_WrongClass_multicolor.png', single_merged)
# 显示图片
cv2.imshow("output", single_merged)
# 不闪退
cv2.waitKey(0)
cv2.destroyAllWindows()

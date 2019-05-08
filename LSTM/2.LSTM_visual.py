import cv2
import random
from scipy.io import *
import pandas as pd
import numpy as np
import tensorflow as tf

'''
    实现基础LSTM网络
'''
# 输入数据是1*100
n_inputs = 100  # 输入一行，一行有100个数据
max_time = 1  # 一共1行
lstm_size = 100  # 隐层单元
n_classes = 10  # 10个分类

# 这里的none表示第一个维度可以是任意的长度
x = tf.placeholder(tf.float32, [None, 100])
# 正确的标签
y = tf.placeholder(tf.float32, [None, 10])

# 初始化权值
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
# 初始化偏置值
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))

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

# 获取test标记onehot矩阵
data_all_label_onehot = pd.read_csv('../dataset/CNN_label_42776_onehot.csv', header=None)
data_all_label_onehot = data_all_label_onehot.as_matrix()
print(data_all_label_onehot.shape)  # (42776, 10)

# 定义LSTM网络
def LSTM(X, weights, biases):
    # inputs=[batch_size, max_time, n_inputs]
    inputs = tf.reshape(X, [-1, max_time, n_inputs])  # 转化数据格式，-1对应一个批次的100
    # 定义LSTM基本CELL单元
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    # final_state[0]是cell_state
    # final_state[1]是hidden_state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results

# 计算LSTM的返回结果
prediction = LSTM(x, weights, biases)
# 损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 把correct_prediction变为float32类型

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
#     saver.restore(sess, './models/LSTM.ckpt')
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
#         acc = sess.run(accuracy, feed_dict={x: band, y: label})
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
#     print(right / 42776)  # 0.7038292500467552 此处的正确率验证上述操作没有问题
#     print(len(pred))
#
#     # pred_label = pd.DataFrame(pred_label)
#     # pred_label.to_csv('../dataset/LSTM_pred_label.csv', header=False, index=False)
#
#     pred = pd.DataFrame(pred)
#     pred.to_csv('../dataset/LSTM_pred.csv', header=False, index=False)

'''
    可视化
'''

# 获取判定结果正误统计
pred = pd.read_csv('../dataset/LSTM_pred.csv', header=None)
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

print(len(wrong))  # 12669 此处验证没有问题，错了30%

'''
    单颜色画图并将错误的分类像素标记LSTM_WrongClass.png
'''

# 读取标记图片paviaU_gt
output_image = loadmat('../dataset/origin/PaviaU_gt.mat')['paviaU_gt']  # (610, 340)

# 单颜色（黄色）显示标记图片

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
# cv2.imwrite('../imgs/LSTM_WrongClass.png', single_merged)
# # 显示图片
# cv2.imshow("output", single_merged)
# # 不闪退
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''
    多颜色画图并将错误的分类像素标记CNN_WrongClass_multicolor.png
'''

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
cv2.imwrite('../imgs/LSTM_WrongClass_multicolor.png', single_merged)
# 显示图片
cv2.imshow("output", single_merged)
# 不闪退
cv2.waitKey(0)
cv2.destroyAllWindows()

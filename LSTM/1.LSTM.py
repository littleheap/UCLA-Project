import random
import pandas as pd
import tensorflow as tf

'''
    实现基础LSTM网络
'''
# 输入数据是1*100
n_inputs = 100  # 输入一行，一行有100个数据
max_time = 1  # 一共1行
lstm_size = 100  # 隐层单元
n_classes = 10  # 10个分类

# 每个批次的大小
batch_size = 100

# 计算一共有多少个训练批次遍历一次训练集
n_batch = 1800 // batch_size  # 18

# 这里的none表示第一个维度可以是任意的长度
x = tf.placeholder(tf.float32, [None, 100])
# 正确的标签
y = tf.placeholder(tf.float32, [None, 10])

# 初始化权值
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
# 初始化偏置值
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


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


'''
    测试集导入
'''
# test测试集
data_test = pd.read_csv('../dataset/CNN_test_40976_shuffle.csv', header=None)
data_test = data_test.as_matrix()

# 获取test特征矩阵
data_test_band = data_test[:, :-1]
print(data_test_band.shape)  # (40976, 103)
# 取前100个通道
data_test_band = data_test_band[:, :100]
print(data_test_band.shape)  # (40976, 100)

# 获取test标记onehot矩阵
data_test_label = pd.read_csv('../dataset/CNN_test_40976_shuffle_label_onehot.csv', header=None)
data_test_label = data_test_label.as_matrix()
print(data_test_label.shape)  # (40976, 10)

'''
    训练集导入
'''
# train训练集
data_train = pd.read_csv('../dataset/CNN_train_1800_shuffle.csv', header=None)
data_train = data_train.as_matrix()

# 获取train特征矩阵
data_train_band = data_train[:, :-1]
print(data_train_band.shape)  # (1800, 103)

# 获取train标记onehot矩阵
data_train_label = pd.read_csv('../dataset/CNN_train_1800_shuffle_label_onehot.csv', header=None)
data_train_label = data_train_label.as_matrix()
print(data_train_label.shape)  # (1800, 10)

'''
    训练网络
'''
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
# 初始化
init = tf.global_variables_initializer()

# 获取100个1800以内的随机数
def get_random_100():
    random_100 = []
    while (len(random_100) < 100):
        x = random.randint(0, 1799)
        if x not in random_100:
            random_100.append(x)
    return random_100

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1000):
        for batch in range(n_batch):
            random_100 = get_random_100()
            batch_xs = data_train_band[random_100][:, :100]
            batch_ys = data_train_label[random_100]
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: data_test_band, y: data_test_label})
        print("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))
        '''
            Iter 339, Testing Accuracy= 0.7063403
            Iter 340, Testing Accuracy= 0.7053153
            Iter 341, Testing Accuracy= 0.7022159
            Iter 342, Testing Accuracy= 0.6991898
            Iter 343, Testing Accuracy= 0.6976523
            Iter 344, Testing Accuracy= 0.70021474
            Iter 345, Testing Accuracy= 0.7029481
            Iter 346, Testing Accuracy= 0.7077802
        '''

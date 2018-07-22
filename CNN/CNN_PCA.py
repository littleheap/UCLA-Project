import random
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA

'''
    实现基础CNN网络
'''
# 每个批次的大小
batch_size = 100

# 计算一共有多少个训练批次遍历一次训练集
n_batch = 1800 // batch_size  # 18

# 制作1800全集验证集，此处与测试集取同，PCA降维至100特征
# 获取特征矩阵
data_band_test = pd.read_csv('./dataset/CNN_data_shuffle.csv', header=None)
data_band_test = data_band_test.as_matrix()
data_band_test = data_band_test[:, :-1]

pca = PCA(n_components=100)
data_band_test = pca.fit_transform(data_band_test)
data_band_test = data_band_test[:, :100]
# print(data_band_test.shape)  # (1800, 100)

# 获取标记onehot矩阵
data_label_test = pd.read_csv('./dataset/CNN_data_shuffle_onehot_label.csv', header=None)
data_label_test = data_label_test.as_matrix()


# print(data_label_test.shape)  # (1800, 10)


# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图


# 初始化权值
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 生成一个截断的正态分布，标准差为0.1
    return tf.Variable(initial, name=name)


# 初始化偏置
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# 卷积层
def conv2d(x, W):
    '''
    x是一个四维的tensor [batch, in_height, in_width, in_channels]
    (1)批次  (2)图片高  (3)图片宽  (4)通道数：黑白为1，彩色为3

    W是一个滤波器/卷积核 [filter_height, filter_width, in_channels, out_channels]
    (1)滤波器高  (2)滤波器宽  (3)输入通道数  (4)输出通道数

    约定 strides[0] = strides[3] = 1， strides[1]代表x方向的步长，strides[2]代表y方向的步长

    padding= 'SAME' / 'VALID' ; SAME在外围适当补0 , VALID不填补0
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2_first(x):
    '''
    x是一个四维的tensor [batch, in_height, in_width, in_channels]
    (1)批次  (2)图片高  (3)图片宽  (4)通道数：黑白为1，彩色为3

    ksize是窗口大小
    约定 ksize[0] = ksize[3] = 1，ksize[1]代表x方向的大小 , ksize[2]代表y方向的大小

    约定 strides[0] = strides[3] = 1，strides[1]代表x方向的步长，strides[2]代表y方向的步长

    padding= 'SAME' / 'VALID' ; SAME在外围适当补0 , VALID不填补0
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2_second(x):
    '''
    x是一个四维的tensor [batch, in_height, in_width, in_channels]
    (1)批次  (2)图片高  (3)图片宽  (4)通道数：黑白为1，彩色为3

    ksize是窗口大小
    约定 ksize[0] = ksize[3] = 1，ksize[1]代表x方向的大小 , ksize[2]代表y方向的大小

    约定 strides[0] = strides[3] = 1，strides[1]代表x方向的步长，strides[2]代表y方向的步长

    padding= 'SAME' / 'VALID' ; SAME在外围适当补0 , VALID不填补0
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')


# 输入层
with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 100], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('x_image'):
        '''
        改变x的格式转为2维的向量 [batch, in_height, in_width, in_channels] 
        (1)批次  (2)二维高  (3)二维宽  (4)通道数：黑白为1，彩色为3
        '''
        x_image = tf.reshape(x, [-1, 10, 10, 1], name='x_image')

# 第一层：卷积+激活+池化
with tf.name_scope('Conv1'):
    # 初始化第一层的W和b
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([2, 2, 1, 32], name='W_conv1')  # 2*2的采样窗口，32个卷积核从1个平面抽取特征
        variable_summaries(W_conv1)
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32], name='b_conv1')  # 每一个卷积核一个偏置值
        variable_summaries(b_conv1)

    # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2_first(h_conv1)  # 进行max-pooling

# 第二层：卷积+激活+池化
with tf.name_scope('Conv2'):
    # 初始化第二个卷积层的权值和偏置
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable([2, 2, 32, 64], name='W_conv2')  # 2*2的采样窗口，64个卷积核从32个平面抽取特征
        variable_summaries(W_conv2)
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')  # 每一个卷积核一个偏置值
        variable_summaries(b_conv2)

    # 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2_second(h_conv2)  # 进行max-pooling

# 10*10的图片第一次卷积后还是10*10，第一次池化后变为5*5
# 第二次卷积后为5*5，第二次池化后变为了4*4
# 进过上面操作后得到64张4*4的平面

with tf.name_scope('fc1'):
    # 初始化第一个全连接层的权值
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([4 * 4 * 64, 1024], name='W_fc1')  # 输入层有4*4*64个列的属性，全连接层有1024个隐藏神经元
        variable_summaries(W_fc1)
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024], name='b_fc1')  # 1024个节点
        variable_summaries(b_fc1)

    # 把第二层的输出扁平化为1维，-1代表任意值
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 64], name='h_pool2_flat')
    # 求第一个全连接层的输出
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(wx_plus_b1)

    # Dropout处理，keep_prob用来表示处于激活状态的神经元比例
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

with tf.name_scope('fc2'):
    # 初始化第二个全连接层
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024, 10], name='W_fc2')  # 输入为1024个隐藏层神经元，输出层为10个数字可能结果
        variable_summaries(W_fc2)
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10], name='b_fc2')
        variable_summaries(b_fc2)
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope('softmax'):
        # 计算输出
        prediction = tf.nn.softmax(wx_plus_b2)

# 交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
                                   name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

# 使用AdamOptimizer进行优化
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 求准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔列表中
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 获取特征矩阵
data_band = pd.read_csv('./dataset/CNN_data_shuffle.csv', header=None)
data_band = data_band.as_matrix()
data_band = data_band[:, :-1]
# PCA降维至100特征
pca = PCA(n_components=100)
data_band = pca.fit_transform(data_band)
# print(data_band.shape)  # (1800, 103)

# 获取标记onehot矩阵
data_label = pd.read_csv('./dataset/CNN_data_shuffle_onehot_label.csv', header=None)
data_label = data_label.as_matrix()


# print(data_label.shape)  # (1800, 10)


# 获取0-1800之间100个随机数用于选取训练集batch
def get_random_100():
    random_100 = []
    while (len(random_100) < 100):
        x = random.randint(0, 1799)
        if x not in random_100:
            random_100.append(x)
    # print(random_100)
    # print(len(random_100))
    return random_100


# 初始化变量
init = tf.global_variables_initializer()

# 合并所有的Summary
merge = tf.summary.merge_all()

# 训练模型存储
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    # 将图写入制定目录
    writer = tf.summary.FileWriter('./logs/CNN_PCA/', sess.graph)
    for i in range(801):
        for batch in range(n_batch):
            # 训练模型
            random_100 = get_random_100()
            batch_xs = data_band[random_100][:, :100]
            batch_ys = data_label[random_100]
            summary, _ = sess.run([merge, train_step],
                                  feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.8})  # dropout比例
        writer.add_summary(summary, i)
        test_acc = sess.run(accuracy, feed_dict={x: data_band_test, y: data_label_test, keep_prob: 1.0})
        print("Training Times：" + str(i) + " , Testing Accuracy = " + str(test_acc))
    # 保存模型
    saver.save(sess, 'net/CNN_PCA/CNN_PCA.ckpt')
'''
    ...
    Training Times：790 , Testing Accuracy = 0.98777777
    Training Times：791 , Testing Accuracy = 0.9872222
    Training Times：792 , Testing Accuracy = 0.98777777
    Training Times：793 , Testing Accuracy = 0.98777777
    Training Times：794 , Testing Accuracy = 0.98777777
    Training Times：795 , Testing Accuracy = 0.98777777
    Training Times：796 , Testing Accuracy = 0.98777777
    Training Times：797 , Testing Accuracy = 0.98777777
    Training Times：798 , Testing Accuracy = 0.98777777
    Training Times：799 , Testing Accuracy = 0.9872222
    Training Times：800 , Testing Accuracy = 0.98777777
'''

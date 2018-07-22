import random
import pandas as pd
import tensorflow as tf

'''
    1.获取9个类别，每个类别200条，一共1800条数据，每条数据103通道信息+一列label，一共(1800,104)
'''
'''
# 导入数据集切割训练与测试数据（用最后处理出的最标准的数据，标准化统一）
data = pd.read_csv('../dataset/PaviaU_gt_band_label_loc_.csv', header=None)
data = data.as_matrix()

# 获取特征矩阵
data_content = data[:, :-2]
# print(data_content.shape)  # (42776, 103)

# 获取标记矩阵
data_label = data[:, -2]
# print(data_label.shape)  # (42776,)

# 切割训练集测试集
data_train, data_test, label_train, label_test = train_test_split(data_content, data_label, test_size=0.3)

# print(data_train.shape)  # (29943, 103)
# print(data_test.shape)  # (12833, 103)
# print(label_train.shape)  # (29943,)
# print(label_test.shape)  # (12833,)

# 获取不同类别指定个数的通道数据，返回该类别二维通道数据，类别label和行数number指定，通道数为103，104通道为label
def get_band(label, n):
    band = []
    for i in range(42776):
        currentband = list(data_content[i])
        currentlabel = data_label[i]
        currentband.append(label)
        if currentlabel == label:
            band.append(currentband)
        if len(list(band)) == n:
            break
    band = list(band)
    # print(len(band))
    band_matrix = np.array(band)
    # print(band.shape)
    # print(band_matrix)
    return band_matrix


data = list()

for i in range(10):
    if i == 0:
        continue
    band_i = get_band(i, 200)
    for j in range(200):
        row = band_i[j, :]
        data.append(row)

new_data = pd.DataFrame(data)
print(new_data.shape)
new_data.to_csv('./dataset/CNN_data.csv', header=False, index=False)
'''

'''
    2.将1800条数据打乱顺序
'''
'''
# 导入数据集切割训练与测试数据（用最后处理出的最标准的数据，标准化统一）
data = pd.read_csv('./dataset/CNN_data.csv', header=None)
data = data.as_matrix()

# 获取特征矩阵
data_content = data[:, :-1]
print(data_content.shape)  # (1800, 103)

# 获取标记矩阵
data_label = data[:, -1]
print(data_label.shape)  # (1800,)

new_csv = []

for i in range(1800):
    new_csv.append(data[i])

random.shuffle(new_csv)
new_csv = pd.DataFrame(new_csv)
new_csv.to_csv('./dataset/CNN_data_shuffle.csv', header=False, index=False)
'''

'''
    3.将标签转化为onehot矩阵格式，并存储
'''
'''
# onehot函数
def onehot(labels, length):
    sess = tf.Session()
    batch_size = tf.size(labels)  # 5
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels], 1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, length]), 1.0, 0.0)
    # print(sess.run(onehot_labels))
    return onehot_labels


# onehot([1, 3, 5, 7, 9], 10)



# 导入乱序数据集切割训练与测试数据（用最后处理出的最标准的数据，标准化统一）
data = pd.read_csv('./dataset/CNN_data_shuffle.csv', header=None)
data = data.as_matrix()

# 获取特征矩阵
data_content = data[:, :-1]
print(data_content.shape)  # (1800, 103)

# 获取标记矩阵
data_label = data[:, -1]
print(data_label.shape)  # (1800,)

label = []

for i in range(1800):
    label.append(int(data_label[i]))

onehot_label = onehot(label, 10)

print(onehot_label)
print(onehot_label.shape)  # (1800, 10)

onehot_csv = []

sess = tf.Session()

for i in range(1800):
    onehot_csv.append(sess.run(onehot_label[i]))

onehot_csv = pd.DataFrame(onehot_csv)
onehot_csv.to_csv('./dataset/CNN_data_shuffle_onehot_label.csv', header=False, index=False)
'''

'''
    4.分别按批次读取通道数据和onehot标记数据过程操作
'''

'''
# 获取特征矩阵
data_band = pd.read_csv('./dataset/CNN_data_shuffle.csv', header=None)
data_band = data_band.as_matrix()
data_band = data_band[:, :-1]
print(data_band.shape)  # (1800, 103)

# 获取标记onehot矩阵
data_label = pd.read_csv('./dataset/CNN_data_shuffle_onehot_label.csv', header=None)
data_label = data_label.as_matrix()
print(data_label.shape)  # (1800, 10)


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


random_100 = get_random_100()
data_band_batch_100 = data_band[random_100]
data_label_batch_100 = data_label[random_100]

# print(data_band_batch_100)
# print(data_band_batch_100.shape)  # (100, 103)

# 取前100个通道，方便输入网络
print(data_band_batch_100[:, :100])
print(data_band_batch_100[:, :100].shape)

# print(data_label_batch_100)
# print(data_label_batch_100.shape)  # (100, 10)
'''

'''
    5.将42776标签转化为onehot矩阵格式，并存储
'''


# onehot函数
def onehot(labels, length):
    sess = tf.Session()
    batch_size = tf.size(labels)  # 5
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels], 1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, length]), 1.0, 0.0)
    # print(sess.run(onehot_labels))
    return onehot_labels


# onehot([1, 3, 5, 7, 9], 10)


# 导入乱序数据集切割训练与测试数据（用最后处理出的最标准的数据，标准化统一）
data = pd.read_csv('../dataset/PaviaU_gt_band_label_loc_.csv', header=None)
data = data.as_matrix()

# 获取特征矩阵
data_content = data[:, :-2]
print(data_content.shape)  # (42776, 103)

# 获取标记矩阵
data_label = data[:, -2]
print(data_label.shape)  # (42776,)

label = []

for i in range(42776):
    label.append(int(data_label[i]))

onehot_label = onehot(label, 10)

print(onehot_label)
print(onehot_label.shape)  # (42776, 10)

print(onehot_label)

onehot_csv = []

# 将tensor转化为numpy
sess = tf.Session()
onehot_label_numpy = onehot_label.eval(session=sess)
print(onehot_label_numpy)
print(onehot_label_numpy.shape)  # (42776, 10)

for i in range(42776):
    onehot_csv.append(onehot_label_numpy[i])

onehot_csv = pd.DataFrame(onehot_csv)
onehot_csv.to_csv('./dataset/CNN_data_42776_onehot_label.csv', header=False, index=False)

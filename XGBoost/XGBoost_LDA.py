import xgboost as xgb
import pandas as pd
from sklearn.linear_model import LogisticRegression


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    # print(acc)
    print(tip + '正确率：\t', float(acc.sum()) / a.size)


if __name__ == '__main__':
    # train训练集
    data_train = pd.read_csv('../CNN/CNN_LDA/dataset/reduce_data_train_shuffle.csv', header=None)
    data_train = data_train.as_matrix()
    # 获取train特征矩阵
    data_train_band = data_train[:, :-1]
    print(data_train_band.shape)  # (1800, 8)
    # 获取train标记label
    data_train_label = data_train[:, -1]

    # test测试集
    data_test = pd.read_csv('../CNN/CNN_LDA/dataset/reduce_data_test_shuffle.csv', header=None)
    data_test = data_test.as_matrix()
    # 获取test特征矩阵
    data_test_band = data_test[:, :-1]
    print(data_test_band.shape)  # (40976, 8)
    # 获取test标记label
    data_test_label = data_test[:, -1]

    x_train = data_train_band
    y_train = data_train_label

    x_test = data_test_band
    y_test = data_test_label

    # Logistic回归
    lr = LogisticRegression(penalty='l2')  # L2正则
    lr.fit(x_train, y_train.ravel())
    y_hat = lr.predict(x_test)
    show_accuracy(y_hat, y_test, 'Logistic回归 ')
    # >>>Logistic回归 正确率：	 0.7933424443576728

    # XGBoost
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 6, 'eta': 0.15, 'silent': 0, 'objective': 'multi:softmax', 'num_class': 10}
    bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)
    y_hat = bst.predict(data_test)
    show_accuracy(y_hat, y_test, 'XGBoost ')
    # >>>XGBoost 正确率：	 0.8534019914096056

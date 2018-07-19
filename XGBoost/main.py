import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    # print(acc)
    print(tip + '正确率：\t', float(acc.sum()) / a.size)


if __name__ == '__main__':
    # 导入数据集切割训练与测试数据（用最后处理出的最标准的数据，标准化统一）
    data = pd.read_csv('../dataset/PaviaU_gt_band_label_loc_.csv', header=None)
    data = data.as_matrix()

    # 获取特征矩阵
    data_band = data[:, :-2]
    # print(data_band.shape)  # (42776, 103)

    # 获取标记矩阵
    data_label = data[:, -2]
    # print(data_label.shape)  # (42776,)

    x_train, x_test, y_train, y_test = train_test_split(data_band, data_label, random_state=1, train_size=0.6)

    # Logistic回归
    lr = LogisticRegression(penalty='l2')  # L2正则
    lr.fit(x_train, y_train.ravel())
    y_hat = lr.predict(x_test)
    show_accuracy(y_hat, y_test, 'Logistic回归 ')  # >>>0.8541873648530185

    # XGBoost
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_test, 'eval'), (data_train, 'train')]
    param = {'max_depth': 6, 'eta': 0.15, 'silent': 0, 'objective': 'multi:softmax', 'num_class': 10}
    bst = xgb.train(param, data_train, num_boost_round=4, evals=watch_list)
    y_hat = bst.predict(data_test)
    show_accuracy(y_hat, y_test, 'XGBoost ')  # >>>0.864122494301911

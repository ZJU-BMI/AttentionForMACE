import csv

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit

from data import read_data, DataSet
from models import BasicLSTMModel


class ExperimentSetup(object):
    kfold = 5
    test_size = 1 / kfold
    train_size = 1 - test_size
    batch_size = 64
    random_state = 1

    lstm_size = 200


def evaluate(tol_label, tol_pred, result_file='resources/save/evaluation_result.csv'):
    """对模型的预测性能进行评估

    :param tol_label: 测试样本的真实标签
    :param tol_pred: 测试样本的预测概率分布
    :param result_file: 结果保存的文件
    :return: 正确率，AUC，精度，召回率， F1值
    """
    assert tol_label.shape == tol_pred.shape
    classes = tol_label.shape[1]

    y_true = np.argmax(tol_label, axis=1)
    y_pred = np.argmax(tol_pred, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(tol_label, tol_pred, average=None)
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f_score = f1_score(y_true, y_pred, average=None)
    with open(result_file, 'a', newline='') as csv_file:
        f_writer = csv.writer(csv_file, delimiter=',')
        for i in range(classes):
            y_score = tol_pred[:, i]
            y_true = tol_label[:, i]
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            f_writer.writerow(f'false positive rate and true positive rate of class {i}')
            f_writer.writerow(fpr)
            f_writer.writerow(tpr)
        f_writer.writerow(accuracy)
        f_writer.writerow(auc)
        f_writer.writerow(precision)
        f_writer.writerow(recall)
        f_writer.writerow(f_score)
        f_writer.writerow([])
    return accuracy, auc, precision, result_file, f_score


def basic_lstm_model_experiments():
    data_set = read_data()
    static_feature = data_set.static_feature
    dynamic_feature = data_set.dynamic_feature
    labels = data_set.labels
    split = StratifiedShuffleSplit(ExperimentSetup.kfold, ExperimentSetup.test_size, ExperimentSetup.train_size)\
        .split(static_feature, labels)

    num_features = dynamic_feature.shape[2]
    time_steps = dynamic_feature.shape[1]
    n_output = labels.shape[1]

    tol_pred = np.zeros(shape=(0, n_output))
    tol_label = np.zeros(shape=(0, n_output), dtype=np.int32)

    model = BasicLSTMModel(num_features,
                           time_steps,
                           ExperimentSetup.batch_size,
                           ExperimentSetup.lstm_size,
                           n_output,
                           epochs=1000)

    for train_idx, test_idx in split:
        train_static = static_feature[train_idx]
        train_dynamic = dynamic_feature[train_idx]
        train_y = labels[train_idx]
        train_set = DataSet(train_static, train_dynamic, train_y)

        model.fit(train_set)

        test_x = dynamic_feature[test_idx]
        test_y = labels[test_idx]

        y_score = model.predict(test_x)
        tol_pred = np.vstack((tol_pred, y_score))
        tol_label = np.vstack((tol_label, test_y))

    result_file = './resources/save/basic_lstm_result.csv'
    acc, auc, precision, recall, f_score = evaluate(tol_label, tol_pred, result_file)
    print(acc)
    print(auc)
    print(precision)
    print(recall)
    print(f_score)


if __name__ == '__main__':
    basic_lstm_model_experiments()

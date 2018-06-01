import csv
import sklearn
import time

import numpy as np
import xlwt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve  # roc计算曲线
from sklearn.model_selection import StratifiedShuffleSplit  # 创建随机数并打乱
import tensorflow as tf

from data import read_data_lu, read_data_sun, DataSet
from models import BasicLSTMModel, BidirectionalLSTMModel, LSTMWithStaticFeature, BiLSTMWithAttentionModel, ResNet, \
    MultiLayerPerceptron


class ExperimentSetup(object):
    kfold = 5
    test_size = 1 / kfold
    train_size = 1 - test_size
    batch_size = 64
    random_state = 1

    lstm_size = 200
    learning_rate = 0.0001
    epochs = 20
    output_n_epochs = 1
    data_source = "lu"


# def evaluate(tol_label, tol_pred, result_file='resources/save/evaluation_result.csv'):
#     """对模型的预测性能进行评估
#
#     :param tol_label: 测试样本的真实标签
#     :param tol_pred: 测试样本的预测概率分布
#     :param result_file: 结果保存的文件
#     :return: 正确率，AUC，精度，召回率， F1值
#     """
#     assert tol_label.shape == tol_pred.shape
#     classes = tol_label.shape[1]
#
#     y_true = np.argmax(tol_label, axis=1)
#     y_pred = np.argmax(tol_pred, axis=1)
#
#     # accuracy = accuracy_score(y_true, y_pred)
#     auc = roc_auc_score(tol_label, tol_pred, average=None)
#     # fpr, tpr, thresholds = roc_curve(y_true, y_score)
#     # precision = precision_score(y_true, y_pred, average=None)
#     # recall = recall_score(y_true, y_pred, average=None)
#     # f_score = f1_score(y_true, y_pred, average=None)
#
#     # auc = roc_auc_score(tol_label[:, 1], tol_pred[:, 1])
#     # # fpr, tpr, thresholds = roc_curve(y_true, y_score)
#     # precision = precision_score(y_true, y_pred)
#     # recall = recall_score(y_true, y_pred)
#     # f_score = f1_score(y_true, y_pred)
#
#     with open(result_file, 'a', newline='') as csv_file:
#         f_writer = csv.writer(csv_file, delimiter=',')
#         # for i in range(classes):
#         #     y_score = tol_pred[:, i]
#         #     y_true = tol_label[:, i]
#         #     fpr, tpr, thresholds = roc_curve(y_true, y_score)
#         # f_writer.writerow('false positive rate and true positive rate of class {}'.format(i))
#         # f_writer.writerow(fpr)
#         # f_writer.writerow(tpr)
#         # f_writer.writerow([accuracy])
#         f_writer.writerow([auc])
#         # f_writer.writerow([precision])
#         # f_writer.writerow([recall])
#         # f_writer.writerow([f_score])
#         # f_writer.writerow([])
#     # return accuracy, auc, precision, recall, f_score
def evaluate(test_index, y_label, y_score, file_name):
    """
    对模型的预测性能进行评估
    :param test_index
    :param y_label: 测试样本的真实标签 true label of test-set
    :param y_score: 测试样本的预测概率 predicted probability of test-set
    :param file_name: 输出文件路径    path of output file
    """
    # TODO 全部算完再写入
    wb = xlwt.Workbook(file_name + '.xls')
    table = wb.add_sheet('Sheet1')
    table_title = ["test_index", "label", "prob", "pre", " ", "fpr", "tpr", "thresholds", " ",
                   "acc", "auc", "recall", "precision", "f1-score", "threshold"]
    for i in range(len(table_title)):
        table.write(0, i, table_title[i])

    auc = roc_auc_score(y_label, y_score)

    fpr, tpr, thresholds = roc_curve(y_label, y_score, pos_label=1)
    threshold = thresholds[np.argmax(tpr - fpr)]

    for i in range(len(fpr)):
        table.write(i + 1, table_title.index("fpr"), fpr[i])
        table.write(i + 1, table_title.index("tpr"), tpr[i])
        table.write(i + 1, table_title.index("thresholds"), float(thresholds[i]))
    table.write(1, table_title.index("threshold"), float(threshold))

    y_pred_label = (y_score >= threshold) * 1
    acc = accuracy_score(y_label, y_pred_label)
    recall = recall_score(y_label, y_pred_label)
    precision = precision_score(y_label, y_pred_label)
    f1 = f1_score(y_label, y_pred_label)

    for i in range(len(test_index)):
        table.write(i + 1, table_title.index("test_index"), int(test_index[i]))
        table.write(i + 1, table_title.index("label"), int(y_label[i]))
        table.write(i + 1, table_title.index("prob"), float(y_score[i]))
        table.write(i + 1, table_title.index("pre"), int(y_pred_label[i]))

    # write metrics
    table.write(1, table_title.index("auc"), float(auc))
    table.write(1, table_title.index("acc"), float(acc))
    table.write(1, table_title.index("recall"), float(recall))
    table.write(1, table_title.index("precision"), float(precision))
    table.write(1, table_title.index("f1-score"), float(f1))

    wb.save(file_name + ".xls")


def model_experiments(model, data_set, result_file):
    static_feature = data_set.static_feature
    dynamic_feature = data_set.dynamic_feature
    labels = data_set.labels
    kf = sklearn.model_selection.StratifiedKFold(n_splits=ExperimentSetup.kfold, shuffle=False)

    n_output = labels.shape[1]  # classes

    tol_test_idx = np.zeros(0, dtype=np.int32)
    tol_pred = np.zeros(shape=(0, n_output))
    tol_label = np.zeros(shape=(0, n_output), dtype=np.int32)
    i = 1
    for train_idx, test_idx in kf.split(X=data_set.dynamic_feature, y=data_set.labels.reshape(-1)):
        train_static = static_feature[train_idx]
        train_dynamic = dynamic_feature[train_idx]
        train_y = labels[train_idx]
        train_set = DataSet(train_static, train_dynamic, train_y)

        test_static = static_feature[test_idx]
        test_dynamic = dynamic_feature[test_idx]
        test_y = labels[test_idx]
        test_set = DataSet(test_static, test_dynamic, test_y)
        print("learning_rate = ", ExperimentSetup.learning_rate)
        model.fit(train_set, test_set)

        y_score = model.predict(test_set)
        tol_test_idx = np.concatenate((tol_test_idx, test_idx))
        tol_pred = np.vstack((tol_pred, y_score))
        tol_label = np.vstack((tol_label, test_y))
        print("Cross validation: {} of {}".format(i, ExperimentSetup.kfold),
              time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        i += 1
        # evaluate(test_y, y_score, result_file)

    model.close()
    # with open(result_file, 'a', newline='') as csv_file:
    #     f_writer = csv.writer(csv_file, delimiter=',')
    #     f_writer.writerow([])
    # return evaluate(tol_label, tol_pred, result_file)
    return evaluate(tol_test_idx, tol_label, tol_pred, result_file)


def basic_lstm_model_experiments(result_file):
    if ExperimentSetup.data_source == 'lu':
        data_set = read_data_lu()
    else:
        data_set = read_data_sun()
    dynamic_feature = data_set.dynamic_feature
    labels = data_set.labels

    num_features = dynamic_feature.shape[2]
    time_steps = dynamic_feature.shape[1]
    n_output = labels.shape[1]

    model = BasicLSTMModel(num_features,
                           time_steps,
                           ExperimentSetup.lstm_size,
                           n_output,
                           batch_size=ExperimentSetup.batch_size,
                           optimizer=tf.train.AdamOptimizer(ExperimentSetup.learning_rate),
                           epochs=ExperimentSetup.epochs,
                           output_n_epoch=ExperimentSetup.output_n_epochs)
    return model_experiments(model, data_set, result_file)


def bidirectional_lstm_model_experiments(result_file):
    if ExperimentSetup.data_source == 'lu':
        data_set = read_data_lu()
    else:
        data_set = read_data_sun()
    dynamic_feature = data_set.dynamic_feature
    labels = data_set.labels

    num_features = dynamic_feature.shape[2]
    time_steps = dynamic_feature.shape[1]
    n_output = labels.shape[1]

    model = BidirectionalLSTMModel(num_features,
                                   time_steps,
                                   ExperimentSetup.lstm_size,
                                   n_output,
                                   batch_size=ExperimentSetup.batch_size,
                                   optimizer=tf.train.AdamOptimizer(ExperimentSetup.learning_rate),
                                   epochs=ExperimentSetup.epochs,
                                   output_n_epoch=ExperimentSetup.output_n_epochs)
    return model_experiments(model, data_set, result_file)


def bi_lstm_attention_model_experiments(result_file, use_attention, use_mlp):
    if ExperimentSetup.data_source == 'lu':
        data_set = read_data_lu()
    else:
        data_set = read_data_sun()
    static_feature = data_set.static_feature
    dynamic_feature = data_set.dynamic_feature
    labels = data_set.labels

    static_n_features = static_feature.shape[1]
    dynamic_n_features = dynamic_feature.shape[2]
    time_steps = dynamic_feature.shape[1]
    n_output = labels.shape[1]

    model = BiLSTMWithAttentionModel(static_n_features,
                                     dynamic_n_features,
                                     time_steps,
                                     ExperimentSetup.lstm_size,
                                     n_output,
                                     use_attention=use_attention,
                                     use_mlp=use_mlp,
                                     batch_size=ExperimentSetup.batch_size,
                                     optimizer=tf.train.AdamOptimizer(ExperimentSetup.learning_rate),
                                     epochs=ExperimentSetup.epochs,
                                     output_n_epoch=ExperimentSetup.output_n_epochs)
    # model_experiments(model, data_set, result_file)
    # with open(result_file, 'a', newline='') as csv_file:
    #     f_writer = csv.writer(csv_file, delimiter=',')
    #     f_writer.writerow([])
    model_experiments(model, data_set, result_file)


def resnet_model_experiments(result_file):
    if ExperimentSetup.data_source == 'lu':
        data_set = read_data_lu()
    else:
        data_set = read_data_sun()
    static_feature = data_set.static_feature
    labels = data_set.labels

    static_n_features = static_feature.shape[1]
    n_output = labels.shape[1]

    model = ResNet(static_n_features,
                   n_output,
                   batch_size=ExperimentSetup.batch_size,
                   optimizer=tf.train.AdamOptimizer(ExperimentSetup.learning_rate),
                   epochs=ExperimentSetup.epochs,
                   output_n_epochs=ExperimentSetup.output_n_epochs)
    return model_experiments(model, data_set, result_file)


def mlp_model_experiment(result_file):
    if ExperimentSetup.data_source == 'lu':
        data_set = read_data_lu()
    else:
        data_set = read_data_sun()
    static_feature = data_set.static_feature
    labels = data_set.labels

    static_n_features = static_feature.shape[1]
    n_output = labels.shape[1]
    model = MultiLayerPerceptron(static_n_features,
                                 ExperimentSetup.lstm_size,
                                 n_output,
                                 batch_size=ExperimentSetup.batch_size,
                                 optimizer=tf.train.AdamOptimizer(ExperimentSetup.learning_rate),
                                 epochs=ExperimentSetup.epochs,
                                 output_n_epoch=ExperimentSetup.output_n_epochs)
    return model_experiments(model, data_set, result_file)


def lstm_with_static_feature_model_experiments(result_file):
    if ExperimentSetup.data_source == 'lu':
        data_set = read_data_lu()
    else:
        data_set = read_data_sun()
    static_feature = data_set.static_feature
    dynamic_feature = data_set.dynamic_feature
    labels = data_set.labels

    static_n_features = static_feature.shape[1]
    dynamic_n_features = dynamic_feature.shape[2]
    time_steps = dynamic_feature.shape[1]
    n_output = labels.shape[1]

    model = LSTMWithStaticFeature(static_n_features,
                                  dynamic_n_features,
                                  time_steps,
                                  ExperimentSetup.lstm_size,
                                  n_output,
                                  batch_size=ExperimentSetup.batch_size,
                                  optimizer=tf.train.AdamOptimizer(ExperimentSetup.learning_rate),
                                  epochs=ExperimentSetup.epochs,
                                  output_n_epochs=ExperimentSetup.output_n_epochs)
    return model_experiments(model, data_set, result_file)


if __name__ == '__main__':
    # basic_lstm_model_experiments('resources/save/basic_lstm.csv')
    # lstm_with_static_feature_model_experiments("resources/save/lstm_with_static.csv")
    # bidirectional_lstm_model_experiments('resources/save/bidirectional_lstm.csv')
    for i_times in range(10):
        # print("res_bi-lstm_att")
        # bi_lstm_attention_model_experiments('result_cx/LAR1-' + str(i_times + 1), True, True)
        # print("bi-lstm_att")
        # bi_lstm_attention_model_experiments('result_cx/LA1-' + str(i_times + 1), True, False)
        print("mlp_bi-lstm")
        bi_lstm_attention_model_experiments('result/ML1-' + str(i_times + 1), False, True)
        # print("bi-lstm")
        # bi_lstm_attention_model_experiments('result_cx/L1-' + str(i_times + 1), False, False)
        # print("resnet")
        # resnet_model_experiments("result/res1-" + str(i_times + 1))
        # print("MLP")
        # mlp_model_experiment("result/MLP1-" + str(i_times + 1))

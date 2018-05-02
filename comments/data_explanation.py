import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer, MinMaxScaler


class DataSet(object):
    def __init__(self, static_feature, dynamic_feature, labels):
        self._static_feature = static_feature
        self._dynamic_feature = dynamic_feature
        self._labels = labels
        self._num_examples = labels.shape[0]  # label：[num_examples, n_output]
        self._epoch_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        if batch_size > self.num_examples:
            raise ValueError('The size of one batch: {} should be less than the total number of '
                             'data: {}'.format(batch_size, self.num_examples))

        start = self._index_in_epoch
        if start + batch_size > self.num_examples:
            self._epoch_completed += 1
            rest_num_examples = self._num_examples - start
            static_rest_part = self._static_feature[start:self._num_examples]
            dynamic_rest_part = self._dynamic_feature[start:self._num_examples]
            label_rest_part = self._labels[start:self._num_examples]

            self._shuffle()  # 打乱
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            static_new_part = self._static_feature[start:end]
            dynamic_new_part = self._dynamic_feature[start:end]
            label_new_part = self._labels[start:end]
            return (np.concatenate((static_rest_part, static_new_part), axis=0),
                    np.concatenate((dynamic_rest_part, dynamic_new_part), axis=0),
                    np.concatenate((label_rest_part, label_new_part), axis=0))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._static_feature[start:end], self._dynamic_feature[start:end], self._labels[start:end]

    def _shuffle(self):
        index = np.arange(self._num_examples)
        np.random.shuffle(index)
        self._static_feature = self.static_feature[index]
        self._dynamic_feature = self.dynamic_feature[index]
        self._labels = self.labels[index]

    @property
    def static_feature(self):
        return self._static_feature

    @property
    def dynamic_feature(self):
        return self._dynamic_feature

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epoch_completed(self):
        return self._epoch_completed

    @epoch_completed.setter
    def epoch_completed(self, value):
        self._epoch_completed = value


class MyScaler(object):

    def __init__(self):
        self.imputer = Imputer()  # 填充
        self.scaler = MinMaxScaler()  # 找出最大值和最小值，进行归一化

    def fit(self, x):
        self.imputer.fit(x) # 处理缺失值，用平均值来填充缺失值
        new_x = self.imputer.transform(x) # 对数据缺失值补全以后，需要转化为Numpy的矩阵格式，方便模型输入
        self.scaler.fit(new_x) 
        return self

    def transform(self, x):
        new_x = self.imputer.transform(x)
        return self.scaler.transform(new_x)

    def fit_trans(self, x):
        return self.fit(x).transform(x)


def read_data_sun():
    static_features = []
    dynamic_features = []
    labels = []
    for i in range(5):
        static_set = pd.read_csv("resources/treatment_5_fold/7Day/s{0}.csv".format(i), index_col=0, encoding='utf-8')
        static_feature = static_set.as_matrix()
        static_features.append(static_feature)

        for j in range(7):
            dynamic_set = pd.read_csv("resources/treatment_5_fold/7Day/d{0}d{1}.csv".format(i, j), index_col=0, encoding="utf-8")
            dynamic_feature = dynamic_set.as_matrix()
            dynamic_features.append(dynamic_feature)

        label_set = pd.read_csv('resources/treatment_5_fold/7Day/m{0}.csv'.format(i), index_col=0, encoding='utf-8')
        label = label_set.as_matrix()
        for k in label:
            if k == ['None']:
                k = [1, 0, 0, 0]
            elif k == ['Ischemia']:
                k = [0, 1, 0, 0]
            elif k == ['Bleeding']:
                k = [0, 0, 1, 0]
            else:
                k = [0, 0, 0, 1]
        # for k in label:
        #     if k == ['None']:
        #         k = [1, 0]
        #     else:
        #         k = [0, 1]
            labels.append(k)
    labels = np.array(labels)
    static_features = np.array(static_features).reshape([-1, 232])
    dynamic_features = np.array(dynamic_features).reshape((7, -1, 2194)).transpose([1, 0, 2])
    return DataSet(static_features, dynamic_features, labels)


def read_data_lu():
    id_set = pd.read_csv("resources/static_features_sorted.csv", encoding='gbk') # 读取文件
    patient_id_list = id_set.iloc[:, 0]  # 得到病人id的list
    static_set = pd.read_csv("resources/data_processed.csv", encoding='gbk') # 读取静态数据
    static_feature = static_set.iloc[:, 12:].as_matrix() # 静态特征

    scaler = MyScaler()
    static_feature = scaler.fit_trans(static_feature)

    dynamic_set = pd.read_csv("resources/treatment_truncated.csv", encoding="utf-8") # 读取动态数据
    max_length = 0
    dynamic_feature = []
    for patient_id in patient_id_list: 
        one_dynamic_feature = dynamic_set.loc[dynamic_set['patient_id'] == patient_id].iloc[:, 3:].as_matrix() # 病人id对应起来，就把特定病人的动态特征读取进去
        if max_length < one_dynamic_feature.shape[0]: # 得到最长的病人的治疗天数
            max_length = one_dynamic_feature.shape[0]
        dynamic_feature.append(one_dynamic_feature)
    # 将病人的天数都补长到max_length
    dynamic_feature = list(map(lambda x: np.pad(x, ((0, max_length-x.shape[0]), (0, 0)), 'constant', constant_values=0),
                               dynamic_feature))
    dynamic_feature = np.stack(dynamic_feature)

    label_set = pd.read_csv('resources/dataset_addmission.csv', encoding='gbk')
    labels = []
    for patient_id in patient_id_list:
        label = label_set.loc[label_set['TraceNo.'] == patient_id].iloc[:, 9:11].as_matrix()
        if np.all(label == [[0, 0]]):  # 无缺血，出血
            label = [1, 0, 0, 0]
        elif np.all(label == [[1, 0]]):  # 缺血
            label = [0, 1, 0, 0]
        elif np.all(label == [[0, 1]]):  # 出血
            label = [0, 0, 1, 0]
        else:
            label = [0, 0, 0, 1]  # 缺血，出血
        # if np.all(label == [[0, 0]]):
        #     label = [1, 0]
        # else:
        #     label = [0, 1]
        labels.append(label)
    labels = np.array(labels)
    return DataSet(static_feature, dynamic_feature, labels)


if __name__ == "__main__":
    read_data_sun()

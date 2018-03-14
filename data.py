import pandas as pd
import numpy as np


class DataSet(object):
    def __init__(self, static_feature, dynamic_feature, labels):
        self._static_feature = static_feature
        self._dynamic_feature = dynamic_feature
        self._labels = labels
        self._num_examples = labels.shape[0]
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

            self._shuffle()
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


def read_data():
    static_set = pd.read_csv("resources/static_features.csv", encoding='gbk')
    static_feature = static_set.iloc[:, 2:].as_matrix()
    patient_id_list = static_set.iloc[:, 0]

    dynamic_set = pd.read_csv("resources/treatment_truncated.csv", encoding="utf-8")
    max_length = 0
    dynamic_feature = []
    for patient_id in patient_id_list:
        one_dynamic_feature = dynamic_set.loc[dynamic_set['patient_id'] == patient_id].iloc[:, 3:].as_matrix()
        if max_length < one_dynamic_feature.shape[0]:
            max_length = one_dynamic_feature.shape[0]
        dynamic_feature.append(one_dynamic_feature)

    dynamic_feature = list(map(lambda x: np.pad(x, ((0, max_length-x.shape[0]), (0, 0)), 'constant', constant_values=0), dynamic_feature))
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
        labels.append(label)
    labels = np.array(labels)
    return DataSet(static_feature, dynamic_feature, labels)

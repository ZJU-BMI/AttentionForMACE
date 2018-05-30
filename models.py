import sklearn

import tensorflow as tf
import numpy as np


def xavier_init(fan_in, fan_out, constant=1):  # 什么意思
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)  # 产生均匀分布随机数，为什么要产生随机分布数？


def residual_block(inputs, num_output, normalizer_params):
    x = tf.contrib.layers.fully_connected(inputs, num_output, normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params)
    x = tf.contrib.layers.fully_connected(x, num_output, normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params)

    origin_dim = inputs.get_shape().as_list()[1]
    if origin_dim == num_output:  # 输出的个数
        return x + inputs
    else:
        residual_weight = tf.Variable(xavier_init(origin_dim, num_output), dtype=tf.float32)
        return x + inputs @ residual_weight


class BasicLSTMModel(object):
    def __init__(self, num_features, time_steps, lstm_size, n_output, batch_size=64, epochs=1000,
                 output_n_epoch=10, optimizer=tf.train.AdamOptimizer(), name='BasicLSTMModel'):
        """

        :param num_features: dimension of input data per time step    time step的作用是什么？
        :param time_steps: max time step ？
        :param batch_size: batch size
        :param lstm_size: size of lstm cell
        :param n_output: classes
        :param epochs: epochs to train
        :param output_n_epoch: output loss per n epoch
        :param optimizer: optimizer
        :param name: model name
        """
        self._num_features = num_features
        self._epochs = epochs
        self._name = name
        self._batch_size = batch_size
        self._output_n_epoch = output_n_epoch
        self._lstm_size = lstm_size

        with tf.variable_scope(self._name):
            self._x = tf.placeholder(tf.float32, [None, time_steps, num_features], 'input')
            self._y = tf.placeholder(tf.float32, [None, n_output], 'label')

            self._hidden_layer()

            self._output = tf.contrib.layers.fully_connected(self._hidden_rep, n_output,
                                                             activation_fn=tf.identity)  # 输出层
            self._pred = tf.nn.softmax(self._output, name="pred")

            self._loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self._y, self._output), name='loss')
            self._train_op = optimizer.minimize(self._loss)

            self._sess = tf.Session()  # 会话

    def _hidden_layer(self):
        lstm = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)  # ??????
        init_state = lstm.zero_state(tf.shape(self._x)[0], tf.float32)  # 全零向量

        mask, length = self._length()  # 每个病人的实际天数
        self._hidden, _ = tf.nn.dynamic_rnn(lstm,
                                            self._x,
                                            sequence_length=length,
                                            initial_state=init_state)
        self._hidden_rep = tf.reduce_sum(self._hidden, 1) / tf.tile(tf.reduce_sum(mask, 1, keepdims=True),
                                                                    (1, self._lstm_size))

    def _length(self):
        mask = tf.sign(tf.reduce_max(tf.abs(self._x), 2))
        length = tf.reduce_sum(mask, 1)
        length = tf.cast(length, tf.int32)
        return mask, length

    def fit(self, data_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0

        logged = set()
        while data_set.epoch_completed < self._epochs:
            _, dynamic_feature, labels = data_set.next_batch(self._batch_size)
            self._sess.run(self._train_op, feed_dict={self._x: dynamic_feature,
                                                      self._y: labels})

            if data_set.epoch_completed % self._output_n_epoch == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                loss = self._sess.run(self._loss, feed_dict={self._x: dynamic_feature,
                                                             self._y: labels})
                print("loss of epoch {} is {}".format(data_set.epoch_completed, loss))

    def predict(self, test_set):
        return self._sess.run(self._pred, feed_dict={self._x: test_set.dynamic_feature})


class BidirectionalLSTMModel(BasicLSTMModel):
    def __init__(self, num_features, time_steps, lstm_size, n_output, batch_size=64, epochs=1000, output_n_epoch=10,
                 optimizer=tf.train.AdamOptimizer(), name='bidirectionalLSTMModel'):
        super().__init__(num_features, time_steps, lstm_size, n_output, batch_size, epochs, output_n_epoch, optimizer,
                         name)

    def _hidden_layer(self):
        self._lstm = {}
        self._init_state = {}
        for direction in ['forward', 'backward']:
            self._lstm[direction] = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)
            self._init_state[direction] = self._lstm[direction].zero_state(tf.shape(self._x)[0], tf.float32)

        mask, length = self._length()
        self._hidden, _ = tf.nn.bidirectional_dynamic_rnn(self._lstm['forward'],
                                                          self._lstm['backward'],
                                                          self._x,
                                                          sequence_length=length,
                                                          initial_state_fw=self._init_state['forward'],
                                                          initial_state_bw=self._init_state['backward'])
        self._hidden_concat = tf.concat(self._hidden, axis=2)  # 沿着num_features的方向进行拼接
        self._hidden_rep = tf.reduce_sum(self._hidden_concat, 1) / tf.tile(tf.reduce_sum(mask, 1, keepdims=True),
                                                                           (1, self._lstm_size * 2))


class LSTMWithStaticFeature(object):
    """同时使用dynamic feature和static feature

    """

    def __init__(self, static_n_features, dynamic_n_feature, time_steps, lstm_size, n_output, batch_size=64,
                 epochs=1000, output_n_epochs=20, optimizer=tf.train.AdamOptimizer(), name="LSTMWithStaticFeature"):
        self._epochs = epochs
        self._name = name
        self._output_n_epochs = output_n_epochs
        self._batch_size = batch_size

        with tf.variable_scope(self._name):
            self._static_input = tf.placeholder(tf.float32, [None, static_n_features])
            self._dynamic_input = tf.placeholder(tf.float32, [None, time_steps, dynamic_n_feature])
            self._y = tf.placeholder(tf.float32, [None, n_output])

            self._lstm = {}
            self._init_state = {}
            for direction in ['forward', 'backward']:
                self._lstm[direction] = tf.contrib.rnn.BasicLSTMCell(lstm_size)
                self._init_state[direction] = self._lstm[direction].zero_state(tf.shape(self._dynamic_input)[0],
                                                                               tf.float32)

            mask, length = self._length()
            self._hidden, _ = tf.nn.bidirectional_dynamic_rnn(self._lstm['forward'],
                                                              self._lstm['backward'],
                                                              self._dynamic_input,
                                                              sequence_length=length,
                                                              initial_state_fw=self._init_state['forward'],
                                                              initial_state_bw=self._init_state['backward'])
            self._hidden_concat = tf.concat(self._hidden, axis=2)
            self._hidden_sum = tf.reduce_sum(self._hidden_concat, 1) / tf.tile(tf.reduce_sum(mask, 1, keep_dims=True),
                                                                               (1, lstm_size * 2))
            self._hidden_rep = tf.concat((self._hidden_sum, self._static_input), axis=1)

            self._output = tf.contrib.layers.fully_connected(self._hidden_rep, n_output,
                                                             activation_fn=tf.identity)
            self._pred = tf.nn.softmax(self._output, name="pred")
            self._loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self._y, self._output), name="loss")
            self._train_op = optimizer.minimize(self._loss)

            self._sess = tf.Session()

    def _length(self):
        mask = tf.sign(tf.reduce_max(tf.abs(self._dynamic_input), 2))
        length = tf.reduce_sum(mask, 1)
        length = tf.cast(length, tf.int32)
        return mask, length

    def fit(self, data_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0

        logged = set()
        while data_set.epoch_completed < self._epochs:
            static_feature, dynamic_feature, labels = data_set.next_batch(self._batch_size)
            self._sess.run(self._train_op, feed_dict={self._static_input: static_feature,
                                                      self._dynamic_input: dynamic_feature,
                                                      self._y: labels})

            if data_set.epoch_completed % self._output_n_epochs == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                loss = self._sess.run(self._loss, feed_dict={self._static_input: static_feature,
                                                             self._dynamic_input: dynamic_feature,
                                                             self._y: labels})
                print("loss of epochs {} is {}".format(data_set.epoch_completed, loss))

    def predict(self, data_set):
        return self._sess.run(self._pred, feed_dict={self._static_input: data_set.static_feature,
                                                     self._dynamic_input: data_set.dynamic_feature})


class BiLSTMWithAttentionModel(object):
    def __init__(self, static_features, dynamic_features, time_steps, lstm_size, n_output, use_attention=True,
                 use_resnet=True, batch_size=64, epochs=1000, output_n_epoch=10, optimizer=tf.train.AdamOptimizer(),
                 name='BasicLSTMModel'):
        self._static_features = static_features
        self._dynamic_features = dynamic_features
        self._time_steps = time_steps
        self._epochs = epochs
        self._name = name
        self._batch_size = batch_size
        self._output_n_epochs = output_n_epoch
        self._lstm_size = lstm_size

        self._use_attention = use_attention
        self._use_resnet = use_resnet

        with tf.variable_scope(self._name):
            self._static_x = tf.placeholder(tf.float32, [None, self._static_features], 'static_input')
            self._dynamic_x = tf.placeholder(tf.float32, [None, time_steps, self._dynamic_features], 'dynamic_input')
            self._is_train = tf.placeholder(tf.bool)
            self._y = tf.placeholder(tf.float32, [None, n_output], 'label')

            self._hidden_layer()

            self._output = tf.contrib.layers.fully_connected(self._hidden_rep, n_output,
                                                             activation_fn=tf.identity)  # 输出层
            self._pred = tf.nn.sigmoid(self._output, name="pred")

            self._loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self._y, self._output), name='loss')
            self._train_op = optimizer.minimize(self._loss)

            self._sess = tf.Session()  # 会话
        self._saver = tf.train.Saver()

    def _hidden_layer(self):
        self._lstm = {}
        self._init_state = {}
        for direction in ['forward', 'backward']:
            self._lstm[direction] = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)
            self._init_state[direction] = self._lstm[direction].zero_state(tf.shape(self._dynamic_x)[0], tf.float32)

        mask, length = self._length()
        self._hidden, _ = tf.nn.bidirectional_dynamic_rnn(self._lstm['forward'],
                                                          self._lstm['backward'],
                                                          self._dynamic_x,
                                                          sequence_length=length,
                                                          initial_state_fw=self._init_state['forward'],
                                                          initial_state_bw=self._init_state['backward'])
        self._hidden_concat = tf.concat(self._hidden, axis=2)  # 沿着num_features的方向进行拼接

        """# 原意正确写法
        if self._use_attention:
            # expand static feature to concat with dynamic feature
            expand_static_x = tf.tile(tf.expand_dims(self._static_x, 1), (1, self._time_steps, 1))
            concat_x = tf.concat((expand_static_x, self._dynamic_x), -1)
            attention_logits = tf.contrib.layers.fully_connected(concat_x, 1, activation_fn=tf.identity)
            self._attention_weight = self._variable_length_softmax(tf.reshape(attention_logits,[-1, self._time_steps]), length)
            # self._hidden_rep = tf.reshape(tf.matmul(tf.transpose(self._attention_weight, [0, 2, 1]),
            #                                         self._hidden_concat), [-1, 2 * self._lstm_size])
            self._hidden_rep = tf.reshape(tf.matmul(tf.reshape(self._attention_weight, [-1, 1, self._time_steps]),
                                                    self._hidden_concat), [-1, 2 * self._lstm_size])"""

        """陈晓芳原版
        if self._use_attention:
            # expand static feature to concat with dynamic feature
            expand_static_x = tf.tile(tf.expand_dims(self._static_x, 1), (1, self._time_steps, 1))
            concat_x = tf.concat((expand_static_x, self._dynamic_x), -1)
            # define the weight and bias to calculate attention weight
            attention_weight = tf.Variable(tf.random_normal((self._static_features + self._dynamic_features, 1), ))
            attention_b = tf.Variable(tf.zeros(1), tf.float32)
            concat_x = tf.reshape(concat_x, (-1, self._static_features + self._dynamic_features))
            weight = concat_x @ attention_weight + attention_b
            weight = tf.reshape(weight, (-1, self._time_steps))  # batch_size * time_steps
            # variable length softmax
            self._attention_weight = self._variable_length_softmax(weight, length)  # length是每个病人的实际天数
            # weighted average the hidden representation, shape of self._hidden is [batch_size, time_steps, lstm_size]
            attention_weight = tf.tile(tf.expand_dims(self._attention_weight, 2), (1, 1, 2 * self._lstm_size))

            self._hidden_rep = tf.reduce_sum(attention_weight * self._hidden_concat, 1)  # 最终隐藏层的表达"""
        # 自己的设想尝试？
        if self._use_attention:
            static_trans = tf.nn.sigmoid(self._static_x)
            static_trans = tf.expand_dims(static_trans, 2)
            dynamic_trans = tf.nn.sigmoid(tf.contrib.layers.fully_connected(self._dynamic_x, self._static_features))
            attention_logits = tf.reshape(tf.matmul(dynamic_trans, static_trans), [-1, self._time_steps])
            self._attention_weight = self._variable_length_softmax(attention_logits, length)
            self._hidden_rep = tf.reshape(tf.matmul(tf.reshape(self._attention_weight, [-1, 1, self._time_steps]),
                                                    self._hidden_concat), [-1, 2 * self._lstm_size])
        else:
            self._hidden_rep = tf.reduce_sum(self._hidden_concat, 1) / tf.tile(tf.reduce_sum(mask, 1, keep_dims=True),
                                                                               (1, self._lstm_size * 2))

        if self._use_resnet:
            self._residual_output = residual_block(self._static_x, 200, {'is_training': self._is_train})
            self._hidden_rep = tf.concat((self._hidden_rep, self._residual_output), 1)

    def _variable_length_softmax(self, logits, length):
        mask = tf.sequence_mask(length, self._time_steps)
        mask_value = tf.as_dtype(tf.float32).as_numpy_dtype(-np.inf) * tf.ones_like(logits)
        mask_logits = tf.where(mask, logits, mask_value)
        return tf.nn.softmax(mask_logits)

    def _length(self):
        mask = tf.sign(tf.reduce_max(tf.abs(self._dynamic_x), 2))
        length = tf.reduce_sum(mask, 1)
        length = tf.cast(length, tf.int32)
        return mask, length

    def fit(self, data_set, test_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0

        logged = set()

        print("auc_qx\tepoch\tloss")
        while data_set.epoch_completed < self._epochs:
            static_feature, dynamic_feature, labels = data_set.next_batch(self._batch_size)
            self._sess.run(self._train_op, feed_dict={self._static_x: static_feature,
                                                      self._dynamic_x: dynamic_feature,
                                                      self._is_train: True,
                                                      self._y: labels})

            if data_set.epoch_completed % self._output_n_epochs == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                loss = self._sess.run(self._loss, feed_dict={self._static_x: static_feature,
                                                             self._dynamic_x: dynamic_feature,
                                                             self._is_train: True,
                                                             self._y: labels})
                y_score = self.predict(test_set)
                # auc_qx = sklearn.metrics.roc_auc_score(test_set.labels[:, 1], y_score[:, 1])
                # auc_cx = sklearn.metrics.roc_auc_score(test_set.labels[:, 2], y_score[:, 2])
                # auc_both = sklearn.metrics.roc_auc_score(test_set.labels[:, 3], y_score[:, 3])
                # print("{}\t{}\t{}\t{}\t{}".format(auc_qx, auc_cx, auc_both, data_set.epoch_completed, loss))
                auc = sklearn.metrics.roc_auc_score(test_set.labels, y_score)
                print("{}\t{}\t{}".format(auc, data_set.epoch_completed, loss))
        self.save(epoch=data_set.epoch_completed)

    def predict(self, data_set):
        return self._sess.run(self._pred, feed_dict={self._static_x: data_set.static_feature,
                                                     self._dynamic_x: data_set.dynamic_feature,
                                                     self._is_train: False})

    def out_attention_weight(self, data_set):
        return self._sess.run(self._attention_weight, feed_dict={self._static_x: data_set.static_feature,
                                                                 self._dynamic_x: data_set.dynamic_feature,
                                                                 self._is_train: False})

    def save(self, path='./model_save/BLAR', epoch=200):
        self._saver.save(self._sess, path, epoch)

    def restore(self, path='./model_save'):
        self._saver.restore(self._sess, tf.train.latest_checkpoint(path))

    def close(self):
        self._sess.close()
        tf.reset_default_graph()


class ResNet(object):
    def __init__(self, static_features, n_output, batch_size=64, epochs=1000, output_n_epochs=10,
                 optimizer=tf.train.AdamOptimizer(), name="ResNet"):
        self._static_features = static_features
        self._epochs = epochs
        self._name = name
        self._batch_size = batch_size
        self._output_n_epochs = output_n_epochs

        with tf.variable_scope(self._name):
            self._static_x = tf.placeholder(tf.float32, [None, self._static_features], 'static_input')
            self._is_training = tf.placeholder(tf.bool)
            self._y = tf.placeholder(tf.float32, [None, n_output], 'label')

            normalizer_params = {'is_training': self._is_training}
            self._out = residual_block(self._static_x, 200, normalizer_params)
            self._out = residual_block(self._out, 200, normalizer_params)
            self._out = residual_block(self._out, 200, normalizer_params)
            self._out = residual_block(self._out, 200, normalizer_params)

            self._output = tf.contrib.layers.fully_connected(self._out, n_output,
                                                             activation_fn=tf.identity)
            self._pred = tf.nn.softmax(self._output, name="pred")

            self._loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self._y, self._output), name='loss')
            self._train_op = optimizer.minimize(self._loss)

            self._sess = tf.Session()

    def fit(self, data_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0

        logged = set()
        while data_set.epoch_completed < self._epochs:
            static_feature, _, labels = data_set.next_batch(self._batch_size)
            self._sess.run(self._train_op, feed_dict={self._static_x: static_feature,
                                                      self._is_training: True,
                                                      self._y: labels})

            if data_set.epoch_completed % self._output_n_epochs == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                pred, loss = self._sess.run((self._pred, self._loss), feed_dict={self._static_x: static_feature,
                                                                                 self._is_training: True,
                                                                                 self._y: labels})
                print("loss of epoch {} is {}".format(data_set.epoch_completed, loss))

    def predict(self, test_set):
        return self._sess.run(self._pred, feed_dict={self._static_x: test_set.static_feature,
                                                     self._is_training: False})

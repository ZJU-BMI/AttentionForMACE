import tensorflow as tf
import numpy as np


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)


class BasicLSTMModel(object):

    def __init__(self, num_features, time_steps, batch_size, lstm_size, n_output, epochs=1000,
                 output_n_epoch=10, optimizer=tf.train.AdamOptimizer(), name='BaiscLSTMModel'):
        """

        :param num_features: dimension of input data per time step
        :param time_steps: max time step
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

        with tf.variable_scope(self._name):
            self._x = tf.placeholder(tf.float32, [None, time_steps, num_features], 'input')
            self._y = tf.placeholder(tf.float32, [None, n_output], 'label')

            lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            init_state = lstm.zero_state(tf.shape(self._x)[0], tf.float32)

            mask, length = self._length()
            self._hidden, _ = tf.nn.dynamic_rnn(lstm,
                                                self._x,
                                                sequence_length=length,
                                                initial_state=init_state)
            self._hidden_sum = tf.reduce_sum(self._hidden, 1) / tf.tile(tf.reduce_sum(mask, 1, keepdims=True), (1, lstm_size))
            self._output = tf.contrib.layers.fully_connected(self._hidden_sum, n_output, activation_fn=tf.identity)
            self._pred = tf.nn.softmax(self._output, name="pred")

            self._loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self._y, self._output), name='loss')
            self._train_op = optimizer.minimize(self._loss)

            self._sess = tf.Session()

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
                loss = self._sess.run(self._loss, feed_dict={self._x: data_set.dynamic_feature,
                                                             self._y: data_set.labels})
                print("loss of epoch {} is {}".format(data_set.epoch_completed, loss))

    def predict(self, dynamic_feature):
        return self._sess.run(self._pred, feed_dict={self._x: dynamic_feature})


class BidirectionalLSTMModel(BasicLSTMModel):
    def __int__(self, num_features, time_steps, lstm_size, batch_size, n_output, epochs=1000,
                output_n_epoch=10, optimizer=tf.train.AdamOptimizer(), name='bidirectional LSTM model'):
        self._epochs = epochs
        self._name = name
        self._batch_size = batch_size
        self._output_n_epoch = output_n_epoch

        with tf.variable_scope(self._name):
            self._x = tf.placeholder(tf.float32, [None, time_steps, num_features])
            self._y = tf.placeholder(tf.float32, [None, n_output], 'label')

            self._lstm = {}
            self._init_state = {}
            for direction in ['forward', 'backward']:
                self._lstm[direction] = tf.contrib.rnn.BasicLSTMCell(lstm_size)
                self._init_state[direction] = self._lstm[direction].zero_state(tf.shape(self._x)[0], tf.float32)

            mask, length = self._length()
            self._hidden, _ = tf.nn.bidirectional_dynamic_rnn(self._lstm['forward'],
                                                              self._lstm['backward'],
                                                              self._x,
                                                              sequence_length=length,
                                                              initial_state_fw=self._init_state['forward'],
                                                              initial_state_bw=self._init_state['backward'])
            self._hidden_concat = tf.concat(self._hidden, axis=2)  # 沿着num_features的方向进行拼接
            self._hidden_sum = tf.reduce_sum(self._hidden_concat) / tf.tile(tf.reduce_sum(mask, 1, keepdims=True), (1, lstm_size * 2))
            self._output = tf.contrib.layers.fuuly_connected(self._hidden_sum, n_output, activation_fn=tf.identity)
            self._pred = tf.nn.softmax(self._output, name="pred")

            self._loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self._y, self._output), name='loss')
            self._train_op = optimizer.minimize(self._loss)

            self._sess = tf.Session()


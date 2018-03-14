import tensorflow as tf
import numpy as np


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high)


class BasicLSTMModel(object):

    def __init__(self, num_features, time_steps, batch_size, lstm_size, n_output, epochs=1000,
                 optimizer=tf.train.AdamOptimizer(), name='BaiscLSTMModel'):
        self._num_features = num_features
        self._epochs = epochs
        self._name = name
        self._batch_size = batch_size

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
            self._hidden_sum = tf.reduce_sum(self._hidden, 1) / tf.tile(tf.reduce_sum(mask, 1, keep_dims=True), (1, 200))
            self._output = tf.contrib.layers.fully_connected(self._hidden_sum, n_output)
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

        while data_set.epoch_completed < self._epochs:
            _, dynamic_feature, labels = data_set.next_batch(self._batch_size)
            self._sess.run(self._train_op, feed_dict={self._x: dynamic_feature,
                                                      self._y: labels})

            if data_set.epoch_completed % 10 == 0:
                loss = self._sess.run(self._loss, feed_dict={self._x: data_set.dynamic_feature,
                                                             self._y: data_set.labels})
                print(f"loss of epoch {data_set.epoch_completed} is {loss}")

    def predict(self, dynamic_feature):
        return self._sess.run(self._pred, feed_dict={self._x: dynamic_feature})

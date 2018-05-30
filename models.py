import tensorflow as tf
import numpy as np

from cells import SRUCell, MySRU
from data import DataSet

__all__ = [
    "BasicLSTMModel",
    "BidirectionalLSTMModel",
    "LSTMWithStaticFeature",
    "BiLSTMWithAttentionModel",
    "ConvolutionModel",
    "ResNet",
    "KnowledgeBaseModel"
]


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
        lstm = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)
        init_state = lstm.zero_state(tf.shape(self._x)[0], tf.float32)  # 全零向量

        mask, length = self._length()  # 每个病人的实际天数
        self._hidden, _ = tf.nn.dynamic_rnn(lstm,
                                            self._x,
                                            sequence_length=length,
                                            initial_state=init_state)
        self._hidden_rep = tf.reduce_sum(self._hidden, 1) / tf.reduce_sum(mask, 1, keepdims=True)

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
        self._hidden_rep = tf.reduce_sum(self._hidden_concat, 1) / tf.reduce_sum(mask, 1, keepdims=True)


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
            self._hidden_sum = tf.reduce_sum(self._hidden_concat, 1) / tf.reduce_sum(mask, 1, keepdims=True)
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
                 name='BiLSTMWithAttentionModel'):
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
            self._pred = tf.nn.softmax(self._output, name="pred")

            self._loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self._y, self._output), name='loss')
            self._train_op = optimizer.minimize(self._loss)

            self._sess = tf.Session()  # 会话

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

        if self._use_attention:
            # expand static feature to concat with dynamic feature
            expand_static_x = tf.tile(tf.expand_dims(self._static_x, 1), (1, self._time_steps, 1))
            concat_x = tf.concat((expand_static_x, self._dynamic_x), -1)
            # define the weight and bias to calculate attention weight
            self._attention_weight = tf.Variable(
                tf.random_normal((self._static_features + self._dynamic_features, 1), ))
            self._attention_b = tf.Variable(tf.zeros(1), tf.float32)
            concat_x = tf.reshape(concat_x, (-1, self._static_features + self._dynamic_features))
            weight = concat_x @ self._attention_weight + self._attention_b
            weight = tf.reshape(weight, (-1, self._time_steps))  # batch_size * time_steps
            # variable length softmax
            attention_weight = self._variable_length_softmax(weight, length)  # length是每个病人的实际天数
            # weighted average the hidden representation, shape of self._hidden is [batch_size, time_steps, lstm_size]
            attention_weight = tf.expand_dims(attention_weight, -1)

            self._hidden_rep = tf.reduce_sum(attention_weight * self._hidden_rep, 1)  # 最终隐藏层的表达
        else:
            self._hidden_rep = tf.reduce_sum(self._hidden_concat, 1) / tf.reduce_sum(mask, 1, keepdims=True)

        if self._use_resnet:
            self._residual_output = residual_block(self._static_x, 200, {'is_training': self._is_train})
            self._hidden_rep = tf.concat((self._hidden_rep, self._residual_output), 1)

    def _variable_length_softmax(self, logits, length):
        mask = tf.sequence_mask(length, self._time_steps)
        mask_value = tf.as_dtype(tf.float32).as_numpy_dtype(-np.inf) * tf.ones_like(logits)
        mask_logits = tf.where(mask, logits, mask_value)
        return tf.nn.softmax(mask_logits)  # [None, time]

    def _length(self):
        mask = tf.sign(tf.reduce_max(tf.abs(self._dynamic_x), 2))
        length = tf.reduce_sum(mask, 1)
        length = tf.cast(length, tf.int32)
        return mask, length

    def fit(self, data_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0

        logged = set()
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
                print("loss of epochs {} is {}".format(data_set.epoch_completed, loss))

    def predict(self, data_set):
        return self._sess.run(self._pred, feed_dict={self._static_x: data_set.static_feature,
                                                     self._dynamic_x: data_set.dynamic_feature,
                                                     self._is_train: False})


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


class ConvolutionModel(object):
    def __init__(self, static_features, dynamic_feature, time_steps, hidden_size, n_class,
                 batch_size=128, epochs=10000, output_n_epochs=20, name="ConvolutionModel"):
        self._static_features = static_features
        self._dynamic_features = dynamic_feature
        self._time_steps = time_steps
        self._hidden_size = hidden_size
        self._n_class = n_class
        self._name = name
        self._batch_size = batch_size
        self._epochs = epochs
        self._output_n_epochs = output_n_epochs

        self._build()

    def _build(self):
        with tf.variable_scope(self._name):
            self._placeholder()
            self._build_set()
            self._conv_layer()
            self._recurrent_layer()
            self._out_layer()
            self._init_sess()

    def _placeholder(self):
        self._static_x = tf.placeholder(tf.float32, [None, self._static_features], 'static_x')
        self._dynamic_x = tf.placeholder(tf.float32, [None, self._time_steps, self._dynamic_features], 'dynamic_x')
        self._y = tf.placeholder(tf.float32, [None, self._n_class], 'labels')

    def _build_set(self):
        with tf.variable_scope('data_set'):
            self._train_set = tf.data.Dataset.from_tensor_slices((self._static_x, self._dynamic_x, self._y))
            self._train_set = self._train_set.repeat().shuffle(self._batch_size * 10).batch(self._batch_size)

            self._test_set = tf.data.Dataset.from_tensor_slices((self._static_x, self._dynamic_x, self._y))
            self._test_set = self._test_set.batch(1)

            self._iter = tf.data.Iterator.from_structure(self._train_set.output_types,
                                                         self._train_set.output_shapes)
            self._static_x_batch, self._dynamic_x_batch, self._y_batch = self._iter.get_next()

            self._switch_train = self._iter.make_initializer(self._train_set)
            self._switch_test = self._iter.make_initializer(self._test_set)

    def _conv_layer(self):
        with tf.variable_scope('conv'):
            self._map_weight = tf.Variable(xavier_init(self._static_features, self._dynamic_features), dtype=tf.float32)
            self._map_bias = tf.Variable(tf.zeros(self._dynamic_features))
            self._static_map = self._static_x_batch @ self._map_weight + self._map_bias
            self._static_map = tf.expand_dims(self._static_map, 1)  # shape (batch, 1, d_features)
            self._concat_feature = tf.concat((self._static_map, self._dynamic_x_batch), 1)
            self._concat_feature = tf.expand_dims(self._concat_feature, -1)  # shape (batch, time_step+1, d_features, 1)

            self._conv_hidden = tf.contrib.layers.conv2d(self._concat_feature,
                                                         self._hidden_size,
                                                         [2, self._dynamic_features],
                                                         padding="VALID")  # (batch, time, 1, 200)
            self._conv_hidden = tf.reshape(self._conv_hidden,
                                           [-1, self._time_steps, self._hidden_size])  # (batch, time, 200)

    def _recurrent_layer(self):
        with tf.variable_scope('recurrent'):
            self._rnn_weight = tf.Variable(xavier_init(self._hidden_size, 3 * self._hidden_size), dtype=tf.float32)
            self._conv_hidden_a = tf.reshape(self._conv_hidden, [-1, self._hidden_size])  # reshape to tensor mut
            self._rnn_input = self._conv_hidden_a @ self._rnn_weight
            self._rnn_input = tf.reshape(self._rnn_input, [-1, self._time_steps, 3 * self._hidden_size])
            self._rnn_input = tf.concat((self._conv_hidden, self._rnn_input), -1)

            self._sru_cell = MySRU(self._hidden_size)
            self._zero_state = self._sru_cell.zero_state(tf.shape(self._dynamic_x_batch)[0], tf.float32)

            mask, length = self._length()
            self._rnn_hidden, _ = tf.nn.dynamic_rnn(cell=self._sru_cell,
                                                    inputs=self._rnn_input,
                                                    sequence_length=length,
                                                    initial_state=self._zero_state)
            self._hidden_rep = tf.reduce_sum(self._rnn_hidden, 1) / tf.reduce_sum(mask, 1, keepdims=True)

    def _out_layer(self):
        with tf.variable_scope("output"):
            self._output = tf.contrib.layers.fully_connected(self._hidden_rep, self._n_class,
                                                             activation_fn=tf.identity)  # 输出层
            self._pred = tf.nn.softmax(self._output, name="pred")

        with tf.variable_scope("loss"):
            self._loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self._y_batch, self._output), name='loss')

        self._train_op = tf.train.AdamOptimizer().minimize(self._loss)

    def _length(self):
        mask = tf.sign(tf.reduce_max(tf.abs(self._dynamic_x_batch), 2))
        length = tf.reduce_sum(mask, 1)
        length = tf.cast(length, tf.int32)
        return mask, length

    def _init_sess(self):
        self._sess = tf.Session()

    def fit(self, data_set: DataSet):
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(self._switch_train, feed_dict={self._static_x: data_set.static_feature,
                                                      self._dynamic_x: data_set.dynamic_feature,
                                                      self._y: data_set.labels})
        for i in range(self._epochs):
            self._sess.run(self._train_op)

            if i % self._output_n_epochs == 0:
                _, loss = self._sess.run((self._train_op, self._loss))
                print(loss)

    def predict(self, data_set: DataSet):
        pred = self._sess.run(self._pred, feed_dict={self._static_x_batch: data_set.static_feature,
                                                     self._dynamic_x_batch: data_set.dynamic_feature,
                                                     self._y_batch: data_set.labels})

        return pred


class KnowledgeBaseModel(object):
    def __init__(self, static_features, dynamic_features, time_steps, lstm_size, n_output, knowledge, batch_size=64, epochs=1000,
                 output_n_epoch=10, optimizer=tf.train.AdamOptimizer(), name='KnowledgeBasedModel'):
        self._static_features = static_features
        self._dynamic_features = dynamic_features
        self._time_steps = time_steps
        self._lstm_size = lstm_size
        self._n_output = n_output
        self._batch_size = batch_size
        self._epochs = epochs
        self._output_n_epoch = output_n_epoch
        self._optimizer = optimizer
        self._name = name

        with tf.variable_scope(self._name):
            self._static_x = tf.placeholder(tf.float32, [None, self._static_features], 'static_input')
            self._dynamic_x = tf.placeholder(tf.float32, [None, time_steps, self._dynamic_features], 'dynamic_input')
            self._is_training = tf.placeholder(tf.bool, name='train_phase')
            self._y = tf.placeholder(tf.float32, [None, n_output], 'label')
            self._knowledge = tf.constant(knowledge, dtype=tf.float32)

            self._hidden_layer()
            self._attention_layer()
            self._concat_layer()

            self._output = tf.contrib.layers.fully_connected(self._hidden_rep, n_output,
                                                             activation_fn=tf.identity)  # 输出层
            self._pred = tf.nn.softmax(self._output, name="pred")

            self._loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self._y, self._output), name='loss')
            self._train_op = optimizer.minimize(self._loss)

            self._sess = tf.Session()  # 会话

    def _hidden_layer(self):
        self._lstm = {}
        self._init_state = {}
        for direction in ['forward', 'backward']:
            self._lstm[direction] = tf.contrib.rnn.BasicLSTMCell(self._lstm_size)
            self._init_state[direction] = self._lstm[direction].zero_state(tf.shape(self._dynamic_x)[0], tf.float32)

        self._mask, self._len = self._length()
        self._hidden, _ = tf.nn.bidirectional_dynamic_rnn(self._lstm['forward'],
                                                          self._lstm['backward'],
                                                          self._dynamic_x,
                                                          sequence_length=self._len,
                                                          initial_state_fw=self._init_state['forward'],
                                                          initial_state_bw=self._init_state['backward'])
        self._hidden_concat = tf.concat(self._hidden, axis=2)  # 沿着num_features的方向进行拼接

    def _attention_layer(self):
        map_weight = tf.Variable(xavier_init(self._dynamic_features, 1), dtype=tf.float32)
        knowledge_map = self._dynamic_x * self._knowledge  # [None, time, f] * [f, ]
        knowledge_map = tf.reshape(knowledge_map, [-1, self._dynamic_features])
        attention_weight = knowledge_map @ map_weight
        attention_weight = tf.reshape(attention_weight, [-1, self._time_steps])
        attention_weight = self._variable_length_softmax(attention_weight, self._len)
        attention_weight = tf.expand_dims(attention_weight, -1)

        self._hidden_rep = tf.reduce_sum(attention_weight * self._hidden_concat, 1)  # 最终隐藏层的表达

    def _concat_layer(self):
        self._residual_output = residual_block(self._static_x, 200, {'is_training': self._is_training})
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

    def fit(self, data_set):
        self._sess.run(tf.global_variables_initializer())
        data_set.epoch_completed = 0

        logged = set()
        while data_set.epoch_completed < self._epochs:
            static_feature, dynamic_feature, labels = data_set.next_batch(self._batch_size)
            self._sess.run(self._train_op, feed_dict={self._static_x: static_feature,
                                                      self._dynamic_x: dynamic_feature,
                                                      self._is_training: True,
                                                      self._y: labels})

            if data_set.epoch_completed % self._output_n_epoch == 0 and data_set.epoch_completed not in logged:
                logged.add(data_set.epoch_completed)
                pred, loss = self._sess.run((self._pred, self._loss), feed_dict={self._static_x: static_feature,
                                                                                 self._dynamic_x: dynamic_feature,
                                                                                 self._is_training: True,
                                                                                 self._y: labels})
                print("loss of epoch {} is {}".format(data_set.epoch_completed, loss))

    def predict(self, test_set):
        return self._sess.run(self._pred, feed_dict={self._static_x: test_set.static_feature,
                                                     self._dynamic_x: test_set.dynamic_feature,
                                                     self._is_training: False})

import tensorflow as tf
from tensorflow.python.layers import base as base_layer


class LayerRNNCell(tf.contrib.rnn.RNNCell):
    def __call__(self, inputs, state, scope=None, *args, **kwargs):
        return base_layer.Layer.__call__(self, inputs, state, scope=scope, *args, **kwargs)


class SRUCell(LayerRNNCell):
    def __init__(self,
                 num_unit,
                 state_is_tuple=True,
                 activation=tf.nn.tanh,
                 reuse=None,
                 name=None,
                 dtype=None):
        super(SRUCell, self).__init__(_reuse=reuse, name=name, dtype=dtype)
        self._num_unit = num_unit
        self._state_is_tuple = state_is_tuple
        self._activation = activation

    @property
    def state_size(self):
        return (tf.contrib.rnn.LSTMStateTuple(self._num_unit, self._num_unit)
                if self._state_is_tuple else 2 * self._num_unit)

    @property
    def output_size(self):
        return self._num_unit

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s" % inputs_shape)
        input_depth = inputs_shape[1].value
        self._kernel = self.add_variable('kernel',
                                         shape=[input_depth, 3 * self._num_unit])
        self._bias = self.add_variable('bias',
                                       shape=[3 * self._num_unit],
                                       initializer=tf.zeros_initializer(dtype=self.dtype))
        self.built = True

    def call(self, inputs, state):
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        gate_inputs = inputs @ self._kernel + self._bias

        i, f, r = tf.split(value=gate_inputs, num_or_size_splits=3, axis=1)
        f = tf.nn.sigmoid(f)
        r = tf.nn.sigmoid(f)
        new_c = f * c + (1 - f) * i
        new_h = r * self._activation(new_c) + (1 - r) * inputs

        if self._state_is_tuple:
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat((new_c, new_h), axis=1)
        return new_h, new_state

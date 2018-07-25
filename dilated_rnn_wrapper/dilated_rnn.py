import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import RNNCell

class DilatedRnnWrapper(RNNCell):
    """ This is wrapper unit should be used like tf.nn.rnn_cell.Residual_wrapper() """
    def __init__(self, cell, dilation_num, batch_size, dtype):
        super(DilatedRnnWrapper, self).__init__()
        self._dilation = dilation_num
        self._time_step = 0
        self._cell = cell
        self.state_list = [self._cell.zero_state(batch_size, dtype) for i in range(self._dilation)]

    @property
    def state_size(self):
        return tf.constant([self._dilation, self._cell.state_size])
    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype):
        return [self._cell.zero_state(batch_size, dtype) for i in range(self._dilation)]

    def call(self, inputs, state):

        new_output, new_state = self._cell(inputs, self.state_list[self._time_step])
        self.state_list[self._time_step] = new_state
        self._time_step += 1
        self._time_step = np.remainder(self._time_step, self._dilation)

        return (new_output, new_state)
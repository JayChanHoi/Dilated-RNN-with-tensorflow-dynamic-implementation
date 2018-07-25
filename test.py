import tensorflow as tf
import numpy as np
from dilated_rnn_wrapper.dilated_rnn import DilatedRnnWrapper

batch_size = 32
num_units = 64
input_dimension =256
seq_len = 16
dtype = tf.float32
dilation_num = 4

input = tf.ones(shape=[batch_size, seq_len, input_dimension], dtype=np.float32)

cell_1 = tf.nn.rnn_cell.GRUCell(num_units=num_units)
initial_state = cell_1.zero_state(batch_size, dtype=tf.float32)

cell_1_dilated = DilatedRnnWrapper(cell=cell_1, dilation_num=dilation_num, batch_size=batch_size, dtype=dtype)

output, state = tf.nn.dynamic_rnn(cell=cell_1_dilated,
                                  inputs=input,
                                  dtype=dtype,
                                  initial_state=initial_state)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    output_ = sess.run([output])
    print("the testing output", output_)
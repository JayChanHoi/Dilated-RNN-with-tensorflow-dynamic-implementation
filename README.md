# Dilated-RNN-with-tensorflow-dynamic-implementation
dilated RNN with dynamic implementation of [Dilated Recurrent Neural Networks](https://arxiv.org/pdf/1710.02224.pdf)

this is a dynamic implemenation of tensorflow. please be noticed that it should be used as tf.nn.rnn_cell.Residual_wrapper.

dilation_num is the width of skip along the time_step dimension. it should be in the formate of 2^n , where n should be any positive integers. 

# advantage of diated rnn
The long term dependancy problem of rnn comes from long time step. dilated rnn reduce this issue by reducing the max length of paths between nodes. But if the skip width is too large, the input features may loss along time dimension.  

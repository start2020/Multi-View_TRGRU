# coding: utf-8
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import libs.model_common

'''
预测目标可以是(M,B,N,N), 也可以是(M,B,N,N,1)
'''
# X=(M,B,N,PN) ,y=(M,B,N,N)
def placeholder(B, P, N, F):
    # x=(M,B,P,N,F) ,y=(M,Bdata_names,N,N), TE=(M,B,P+Q,2)
    samples = tf.compat.v1.placeholder(shape = (None, P, N,F), dtype = tf.float32,name="samples")
    labels = tf.compat.v1.placeholder(shape = (None, N, F), dtype = tf.float32,name="lables")
    return labels,samples

class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
  def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=False, peephole=False, data_format='channels_last', reuse=None):
    super(ConvLSTMCell, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._peephole = peephole
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, state):
    c, h = state

    x = tf.concat([x, h], axis=self._feature_axis)
    n = x.shape[-1].value
    m = 4 * self._filters if self._filters > 1 else 4
    W = tf.get_variable('kernel_{}_{}'.format(n,m), self._kernel + [n, m])
    y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)

    if not self._normalize:
      y += tf.get_variable('bias_{}_{}'.format(n,m), [m], initializer=tf.zeros_initializer())
    j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

    if self._peephole:
      i += tf.get_variable('W_ci_{}_{}'.format(n,m), c.shape[1:]) * c
      f += tf.get_variable('W_cf_{}_{}'.format(n,m), c.shape[1:]) * c

    if self._normalize:
      j = tf.contrib.layers.layer_norm(j)
      i = tf.contrib.layers.layer_norm(i)
      f = tf.contrib.layers.layer_norm(f)

    f = tf.sigmoid(f + self._forget_bias)
    i = tf.sigmoid(i)
    c = c * f + i * self._activation(j)

    if self._peephole:
      o += tf.get_variable('W_co_{}_{}'.format(n,m), c.shape[1:]) * c

    if self._normalize:
      o = tf.contrib.layers.layer_norm(o)
      c = tf.contrib.layers.layer_norm(c)

    o = tf.sigmoid(o)
    h = o * self._activation(c)

    state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
    return h, state

# ConvLSTM核心本质还是和LSTM一样，将上一层的输出作下一层的输入。
# 不同的地方在于加上卷积操作之后，为不仅能够得到时序关系，还能够像卷积层一样提取特征，提取空间特征。为什么卷积能提取空间特征?
# 这样就能够得到时空特征。并且将状态与状态之间的切换也换成了卷积计算。
def Model(std, mean, X,N):
    shape = [N, N]
    kernel = [3, 3]
    filters = [8,8,1]
    inputs = tf.expand_dims(X,axis=-1) # (B,P,N,N,1)
    Convlstm_cell = ConvLSTMCell(shape, filters[0], kernel)
    outputs, state = tf.nn.dynamic_rnn(Convlstm_cell, inputs, dtype=inputs.dtype) #outputs (B,P,N,N,8),c = state[0] (B,N,N),h = state[1] (B,N,N)
    Convlstm_cell_2 = ConvLSTMCell(shape, filters[1], kernel)
    outputs, state = tf.nn.dynamic_rnn(Convlstm_cell_2, outputs, dtype=outputs.dtype)
    Convlstm_cell_3 = ConvLSTMCell(shape, filters[2], kernel)
    outputs, state = tf.nn.dynamic_rnn(Convlstm_cell_3, outputs, dtype=outputs.dtype)
    output = tf.squeeze(outputs[:,-1,...],axis=-1) # 单取它会让TEST没变化，就是预测效果很差
    output = libs.model_common.multi_fc(output, activations="relu", units=[256], drop_rate=None, bn=False, dims=None,
                                   is_training=True)

    # output = libs.model_common.multi_fc(output, activations="relu", units=[256], drop_rate=None, bn=False, dims=None,
    #                                is_training=True)
    outputs = libs.model_common.multi_targets(output, std, mean, N)
    return outputs
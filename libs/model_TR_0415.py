import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def placeholder(Q, P, N, F):
    # x=(M,B,P,N,F) ,y=(M,B,N,N), TE=(M,B,P+Q,2)
    X = tf.compat.v1.placeholder(shape = (None, P, N,F), dtype = tf.float32)
    TE = tf.compat.v1.placeholder(shape = (None, P + Q, 2), dtype = tf.int32)
    label = tf.compat.v1.placeholder(shape = (None, N, F), dtype = tf.float32)
    is_training = tf.compat.v1.placeholder(shape = (), dtype = tf.bool)
    return X, TE, label, is_training

'''
功能: 多个层，每层 =一个dropout(可选) => 卷积 (形状不变，但维度变化)
inputs: (B,T,N,F1)
outputs:(B,T,N,F2)
F: 1,64, 128
用处：是个原子操作，其他组件基本都用它构建。
dropout层只有在最后的decoder才能用到
use_bias 基本上每个地方都用到
activation一般用relu激活，要不就是不用
一般用两层fc，两层fc一般是(relu, none)型的；
而query,value,key用一层fc，用relu激活；
gatefusion的用一层fc，用relu激活
'''
def FC(x, units, activations, bn, bn_decay, is_training, use_bias=True, drop=None):
    # 将输入 units 变成 list
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    elif isinstance(units, tuple):
        units = list(units)
        activations = list(activations)
    assert type(units) == list

    for num_unit, activation in zip(units, activations):
        if drop is not None:
            x = dropout(x, drop=drop, is_training=is_training)
        x = conv2d(
            x, output_dims=num_unit, kernel_size=[1, 1], stride=[1, 1],
            padding='VALID', use_bias=use_bias, activation=activation,
            bn=bn, bn_decay=bn_decay, is_training=is_training)
    return x

'''
功能：
参数：
(1) SE: (N,D)=每个节点都有一个D维的特征，该特征是其空间特征
(2) TE: (B,P+Q,2)=每个样本的都有一个时间特征，该时间特征包含了dayofweek, timeofday
(3) T:每天划分成时间段的个数
(4) D:每个点的时空特征维度
(5) bn:
(6) bn_decay:
(7) is_training:
返回值：(B, P + Q, N, D)
'''
def STEmbedding(SE, TE, T, D, bn, bn_decay, is_training):
    # spatial embedding
    SE = tf.expand_dims(tf.expand_dims(SE, axis = 0), axis = 0) # (1,1,N,F)
    SE = FC(
        SE, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)# (1,1,N,D)

    # temporal embedding
    dayofweek = tf.one_hot(TE[..., 0], depth = 7) # (B, P + Q, 7)
    timeofday = tf.one_hot(TE[..., 1], depth = T) # (B, P + Q, T)
    TE = tf.concat((dayofweek, timeofday), axis = -1)  # (B, P + Q, T+7)
    TE = tf.expand_dims(TE, axis = 2)  # (B, P + Q, 1, T+7)
    TE = FC(
        TE, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)# (B, P + Q, 1, D)
    result =tf.add(SE, TE) # (B, P + Q, N, D)
    return result

'''
参数:
(0) X:(B,P,N,F), F>N
(1) SE: (N,F)=每个节点都有一个F维的特征，该特征是其空间特征
(2) TE: (B,P+Q,2)=每个样本的都有一个时间特征，该时间特征包含了dayofweek, timeofday
(3) T:每天划分成时间段的个数
(4) D:每个点的时空特征维度,D=K*d,K=注意力头数，d=每头输出维度
返回值：(B,N,N)
'''
def Model(mean, std, X, SE, TE, T, K, d, is_training, bn=False, bn_decay=None):
    D = K * d
    P = X.shape[1]
    N = X.shape[2]
    STE = STEmbedding(SE, TE, T, D, bn, bn_decay, is_training)
    STE_P = STE[:, : P] # (B,P,N,K*d)
    # (B,P,N,F) => (B,P,N,K*d)
    X = FC(
        X, units=[D, D], activations=[tf.nn.relu, None],
        bn=bn, bn_decay=bn_decay, is_training=is_training)
    # X=(B,P,N,K*d), STE_P=(B,P,N,K*d) => (B,P,N,2K*d)
    X = tf.concat((X, STE_P), axis = -1)
    # (B,P,N,2K*d) => (B,P,N,K*d)
    query = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # (B,P,N,2K*d) => (B,P,N,K*d)
    key = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # (B,P,N,2K*d) => (B,P,N,K*d)
    value = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # (B,P,N,K*d)=>(BK,P,N,d)
    query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
    key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
    value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
    # (BK,P,N,d)*(BK,P,N,d)=>(BK,P,N,N)
    attention = tf.matmul(query, key, transpose_b = True)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis = -1)
    # (BK,P,N,N) * (BK,P,N,d)=> (BK,P,N,d)
    X = tf.matmul(attention, value)
    # (BK,P,N,d) => (B,P,N,Kd)
    X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
    # (B,P,N,K*d) => (B,P,N,N)
    X = FC(
        X, units = [D, N], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # 简单的fc
    X = tf.transpose(X,[0,2,1,3]) #(B,N,P,N)
    N, P = X.shape[1], X.shape[2]
    X = tf.reshape(X, [-1,N,P*N]) #(B,N,PN)
    X = fc(X,N) # (B,N,N)
    X = X * std + mean
    X = tf.nn.relu(fc(X,N))
    return X

def fc(X, output_unit):
    w = tf.Variable(
        tf.glorot_uniform_initializer()(shape = [X.shape[-1], output_unit]),
        dtype = tf.float32, trainable = True, name = 'w')
    b = tf.Variable(
        tf.glorot_uniform_initializer()(shape = [output_unit]),
        dtype = tf.float32, trainable = True, name = 'b')
    y = tf.matmul(X,w) + b
    return y


############################################### module #####################################
'''
功能：卷积 => 偏置(可选)(是) => 标准化(可选)(否) => 激活(可选)(relu)
输入:[..., input_dim], 输出[..., output_dim]
'''
def conv2d(x, output_dims, kernel_size, stride=[1, 1],
           padding='SAME', use_bias=True, activation=tf.nn.relu,
           bn=False, bn_decay=None, is_training=None):
    input_dims = x.get_shape()[-1].value
    kernel_shape = kernel_size + [input_dims, output_dims]

    # 卷积核用glorot_uniform初始化
    kernel = tf.Variable(
        tf.glorot_uniform_initializer()(shape=kernel_shape),
        dtype=tf.float32, trainable=True, name='kernel')
    # 卷积
    x = tf.nn.conv2d(x, kernel, [1] + stride + [1], padding=padding)
    # 是否使用偏置项
    if use_bias:
        bias = tf.Variable(
            tf.zeros_initializer()(shape=[output_dims]),
            dtype=tf.float32, trainable=True, name='bias')
        x = tf.nn.bias_add(x, bias)
    # 若激活，则可使用batch_norm项
    if activation is not None:
        if bn:
            x = batch_norm(x, is_training=is_training, bn_decay=bn_decay)
        x = activation(x)
    return x

# 层标准化
def batch_norm(x, is_training, bn_decay):
    input_dims = x.get_shape()[-1].value
    moment_dims = list(range(len(x.get_shape()) - 1))
    beta = tf.Variable(
        tf.zeros_initializer()(shape=[input_dims]),
        dtype=tf.float32, trainable=True, name='beta')
    gamma = tf.Variable(
        tf.ones_initializer()(shape=[input_dims]),
        dtype=tf.float32, trainable=True, name='gamma')
    batch_mean, batch_var = tf.nn.moments(x, moment_dims, name='moments')

    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    ema_apply_op = tf.cond(
        is_training,
        lambda: ema.apply([batch_mean, batch_var]),
        lambda: tf.no_op())

    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(
        is_training,
        mean_var_with_update,
        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    x = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return x


# 训练时，采用dropout，测试不采用
def dropout(x, drop, is_training):
    x = tf.cond(
        is_training,
        lambda: tf.nn.dropout(x, rate=drop),
        lambda: x)
    return x

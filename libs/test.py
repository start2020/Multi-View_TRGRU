# coding=gbk
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
X = np.array([6,5,0,9,1,2])
print(X[:-3])
#测试
# B,T,N,F = 32,10,118,2
# L1 = tf.random.normal(shape=(N,N))
# L2 = tf.random.normal(shape=(B,T,N,N))
# X1 = tf.random.normal(shape=(B,N,F))
# X2 = tf.random.normal(shape=(B,T,N,F))
# #output = multi_gcn(L1, X2, activations=["relu","relu"], units=[64,64], Ks = [None,None])
# output = multi_fc(X2, activations=["relu","relu"], units=[64,64])
# print(output.shape)

# 输入
# mat; 矩阵(D,T,N,N),nd.array类型,元素数据类型是int or float32
# start_val: 区间的起始值,区间是闭区间[start,end]
# end_val: 区间的结束值
# 返回值: 这个区间在整个矩阵中占得比例
#
# def get_mat_val(mat, start_val, end_val):
#     mat = np.around(mat)
#     days, T, N, _ = mat.shape
#     sum = 0
#     for i in range(start_val, end_val+1):
#         sum += np.sum(mat == i)
#     total = days*T*N*N
#     proportion = sum/total
#     return proportion
#
# def batch_norm(x, dims, is_training):
#     # 形状
#     shape = x.get_shape().as_list()[-dims:]
#     # 偏置系数，初始值为0
#     beta = tf.Variable(
#         tf.zeros_initializer()(shape=shape),
#         dtype=tf.float32, trainable=True, name='beta')
#     # 缩放系数，初始值为1
#     gamma = tf.Variable(
#         tf.ones_initializer()(shape=shape),
#         dtype=tf.float32, trainable=True, name='gamma')
#     # 计算均值和方差
#     moment_dims = list(range(len(x.get_shape()) - dims))
#     batch_mean, batch_var = tf.nn.moments(x, moment_dims, name='moments')
#
#     # 滑动平均
#     ema = tf.train.ExponentialMovingAverage(0.9)
#     # Operator that maintains moving averages of variables.
#     ema_apply_op = tf.cond(
#         is_training,
#         lambda: ema.apply([batch_mean, batch_var]),
#         lambda: tf.no_op())
#
#     # Update moving average and return current batch's avg and var.
#     def mean_var_with_update():
#         with tf.control_dependencies([ema_apply_op]):
#             return tf.identity(batch_mean), tf.identity(batch_var)
#
#     # ema.average returns the Variable holding the average of var.
#     mean, var = tf.cond(
#         is_training,
#         mean_var_with_update,
#         lambda: (ema.average(batch_mean), ema.average(batch_var)))
#
#     # 标准化:x输入，mean均值，var方差，beta=偏移值，gama=缩放系数，1e-3=防止除零
#     x = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
#     return x
#
#
# def exacute1():
#     is_training = tf.compat.v1.placeholder(shape = (), dtype = tf.bool)
#     B,P,N,N = 32,3,118,118
#     is_training = tf.constant(True)
#     print(is_training)
#     dims = 1
#     x = tf.random.normal(shape=(B,P,N,N))
#     y = batch_norm(x, dims, is_training)
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#     normal_y = sess.run(y,feed_dict={is_training:True})
#     normal_y = tf.constant(normal_y)
#     batch_mean, batch_var = tf.nn.moments(normal_y, [0,1,2], name='moments')
#     mean, var = sess.run([batch_mean, batch_var])
#     print(mean.shape)
#     print(var.shape)
#
# exacute1()
# mat = np.around(np.array([[0.01,2.02,2,3],[4,2,3,5]]))
# days = 2
# T = 3
# N = 4
# mat = np.around(np.random.rand(days,T,N,N)*10)
# print(mat)
#
# min_val = np.min(mat)
# max_val = np.max(mat)
# print(min_val,max_val)
# res = get_mat_val(mat,1,5)
# print(res)

# graph_common

# # semantic graph
# A = np.random.normal(size=(1,3,3))
# M = graph_common.semantic_graph(A)
# print(A)
# print(M)
# MS = graph_common.normalize_attention(M, axis=0)
# MS = graph_common.self_loop(MS)
# print(MS)

# A = np.array([[2,4],[3,6],[4,8]])
# attention = softmax_attention(A, axis=0)
# print(attention)

# 加入自环
# A = np.random.normal(size=(1,2,2))
# print(A.shape)
# print(A)
# print(self_loop(A))

#去掉对角线
# A = np.random.normal(size=(2,2,2))
# M = del_diag(A)
# print(M)

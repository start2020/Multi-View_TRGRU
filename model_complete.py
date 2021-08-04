import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from libs import model_common

'''
labels: 
(B,P,N,N) samples_all_C
inputs: 
(B,P,N,N), samples_P_before_C
(B,P,N,N), samples_P_after_W_C
(B,P,N,N,3), samples_P_before_odt_C
(B,P,N),IN
'''
def placeholder(P, N, F_in, F_out,odt_in):

    labels = tf.compat.v1.placeholder(shape = (None, P, N, F_out), dtype = tf.float32,name="labels")
    samples_P_before_C = tf.compat.v1.placeholder(shape = (None, P, N, F_in), dtype = tf.float32,name="samples_P_before_C")
    samples_P_after_W_C = tf.compat.v1.placeholder(shape = (None, P, N, F_in), dtype = tf.float32,name="samples_P_after_W_C")
    samples_P_before_odtR1_C = tf.compat.v1.placeholder(shape = (None, P, N, F_in, odt_in), dtype = tf.float32,name="samples_P_before_odtR1_C")
    samples_P_before_odtD1_C = tf.compat.v1.placeholder(shape=(None, P, N, F_in, odt_in), dtype=tf.float32,
                                                      name="samples_P_before_odtD1_C")
    IN = tf.compat.v1.placeholder(shape = (None, P, N), dtype = tf.float32,name="IN")
    return labels, samples_P_before_C, samples_P_after_W_C, samples_P_before_odtR1_C,samples_P_before_odtD1_C,IN

# def placeholder(P, N, F_in, F_out,odt_in):
#
#     labels = tf.compat.v1.placeholder(shape = (None, P, N, F_out), dtype = tf.float32,name="labels")
#     samples_P_before_C = tf.compat.v1.placeholder(shape = (None, P, N, F_in), dtype = tf.float32,name="samples_P_before_C")
#     samples_P_after_W_C = tf.compat.v1.placeholder(shape = (None, P, N, F_in), dtype = tf.float32,name="samples_P_after_W_C")
#     samples_P_before_odt_C = tf.compat.v1.placeholder(shape = (None, P, N, F_in, odt_in), dtype = tf.float32,name="samples_P_before_odt_C")
#     IN = tf.compat.v1.placeholder(shape = (None, P, N), dtype = tf.float32,name="IN")
#     return labels, samples_P_before_C, samples_P_after_W_C, samples_P_before_odt_C,IN

def model_1(samples_P_after_W_C):
    SUM = tf.expand_dims(tf.reduce_sum(samples_P_after_W_C,axis=-1),axis=-1) #(B,P,N)=>(B,P,N,1)
    SUM = tf.where(tf.equal(SUM, 0.0), tf.ones_like(SUM), SUM)
    # 如果最后一个维度的值是0，则用1代替，但如果标准化数据后，倒不必不会是0
    P = samples_P_after_W_C / SUM #(B,P,N,N)

    # N = samples_P_after_W_C.get_shape()[-2].value
    # P = model_common.multi_fc(P, activations="relu", units=[N], drop_rate=None, bn=False, dims=None,
    #                                    is_training=True)
    return P

def model_2(samples_P_before_odt_C,IN):
    SUM = tf.expand_dims(tf.reduce_sum(samples_P_before_odt_C,axis=-2),axis=-2) #(B,P,N,3)=>(B,P,N,1,3)
    SUM = tf.where(tf.equal(SUM, 0.0), tf.ones_like(SUM), SUM)
    #如果最后一个维度的值是0，则用1代替，但如果标准化数据后，倒不必不会是0
    P = samples_P_before_odt_C / SUM #(B,P,N,N,3)/(B,P,N,1,3) = (B,P,N,N,3)
    all_pred = tf.expand_dims(tf.expand_dims(IN, axis=-1),axis=-1) * P #(B,P,N,1,1)*(B,P,N,N,3)
    all_preds = tf.zeros_like(all_pred[...,0]) #(B,P,N,N)
    for i in range(3):
        w = tf.Variable(tf.glorot_uniform_initializer()(shape=()),
                        dtype=tf.float32, trainable=True, name='weight%s'%i)  # (F1,)
        all_preds += all_pred[...,i]*w
    return all_preds #(B,P,N,N)
#
# def Model(samples_P_before_C, samples_P_after_W_C, samples_P_before_odt_C,IN, std, mean, F_out):
#     w = tf.Variable(tf.glorot_uniform_initializer()(shape=()),
#                     dtype=tf.float32, trainable=True, name='weight')
#     outputs = model_1(samples_P_before_C, samples_P_after_W_C, IN)*w + model_2(samples_P_before_odt_C,IN)*(1-w)
#     # outputs = model_1(samples_P_before_C, samples_P_after_W_C, IN) * w
#
#     # outputs = model_common.multi_targets(outputs, std, mean, F_out)
#     return outputs



# def model_2(samples_P_before_odt_C,IN):
#     SUM = tf.expand_dims(tf.reduce_sum(samples_P_before_odt_C,axis=-2),axis=-2) #(B,P,N,3)=>(B,P,N,1,3)
#     SUM = tf.where(tf.equal(SUM, 0.0), tf.ones_like(SUM), SUM)
#     #如果最后一个维度的值是0，则用1代替，但如果标准化数据后，倒不必不会是0
#     P = samples_P_before_odt_C / SUM #(B,P,N,N,3)/(B,P,N,1,3) = (B,P,N,N,3)
#     all_pred = tf.expand_dims(tf.expand_dims(IN, axis=-1),axis=-1) * P #(B,P,N,1,1)*(B,P,N,N,3)
#     all_preds = tf.zeros_like(all_pred[...,0]) #(B,P,N,N)
#     N = all_preds.get_shape()[-1].value
#     for i in range(3):
#         w = tf.Variable(tf.glorot_uniform_initializer()(shape=(N, 1)),
#                         dtype=tf.float32, trainable=True, name='weight%s'%i)  # (F1,)
#         all_preds += all_pred[...,i]*w
#     return all_preds #(B,P,N,N)
#
# def Model(samples_P_before_C, samples_P_after_W_C, samples_P_before_odt_C,IN, std, mean, F_out):
#     N = samples_P_before_C.get_shape()[-1].value
#     w = tf.Variable(tf.glorot_uniform_initializer()(shape=(N, 1)),
#                     dtype=tf.float32, trainable=True, name='weight')
#     # w = tf.nn.sigmoid(w)
#     outputs = model_1(samples_P_before_C, samples_P_after_W_C, IN)*w + model_2(samples_P_before_odt_C,IN)*(1-w)
#     # outputs = model_1(samples_P_before_C, samples_P_after_W_C, IN)*w
#     # outputs = model_2(samples_P_before_odt_C, IN)*w
#
#     # outputs = model_common.multi_fc(outputs, activations="relu", units=[288], drop_rate=None, bn=False, dims=None,
#     #                                is_training=True)
#
#     In_preds = tf.reduce_sum(outputs, axis=-1)  # (M,P,N) #这个没用的,只是为了代码可以运行而已
#     Out_preds = tf.reduce_sum(outputs, axis=-2)  # (M,P,N) #这个没用的,只是为了代码可以运行而已
#     # outputs = model_common.multi_targets(outputs, std, mean, F_out) #在output后面加一个fc的话效果更差
#     return [outputs,In_preds,Out_preds]





def model_4(samples_P_before_odtR1_C,samples_P_before_odtW1_C):

    SUM = tf.expand_dims(tf.reduce_sum(samples_P_before_odtR1_C,axis=-2),axis=-2) #(B,P,N,3)=>(B,P,N,1,3)
    SUM = tf.where(tf.equal(SUM, 0.0), tf.ones_like(SUM), SUM)
    #如果最后一个维度的值是0，则用1代替，但如果标准化数据后，倒不必不会是0
    P1 = samples_P_before_odtR1_C / SUM #(B,P,N,N,1)/(B,P,N,1,1) = (B,P,N,N,1)

    SUM = tf.expand_dims(tf.reduce_sum(samples_P_before_odtW1_C, axis=-2), axis=-2)  # (B,P,N,3)=>(B,P,N,1,3)
    SUM = tf.where(tf.equal(SUM, 0.0), tf.ones_like(SUM), SUM)
    # 如果最后一个维度的值是0，则用1代替，但如果标准化数据后，倒不必不会是0
    W1 = samples_P_before_odtW1_C / SUM  # (B,P,N,N,3)/(B,P,N,1,1) = (B,P,N,N,1)

    N = samples_P_before_odtW1_C.get_shape()[-2].value

    proportion = P1[...,0]-W1[...,0]
    print(proportion)
    proportion = model_common.multi_fc(proportion, activations="relu", units=[N], drop_rate=None, bn=False, dims=None,
                                   is_training=True)

    # # proportion = P1[..., 0] - W1[..., 0]
    # proportions = tf.zeros_like(proportion)  # (B,P,N,N)
    # # print(proportions)
    # for i in range(3):
    #     w = tf.Variable(tf.glorot_uniform_initializer()(shape=()),
    #                     dtype=tf.float32, trainable=True, name='weight%s' % i)  # (F1,)
    #     res = proportion[..., i] * w
    #     # print(res)
    #     # proportions[] = res
    #     proportions += res
    # # w = tf.Variable(tf.glorot_uniform_initializer()(shape=(N, 1)),
    # #                     dtype=tf.float32, trainable=True, name='weight%s'%0)  # (F1,)
    # print(proportions)
    

    return proportion #(B,P,N,N)

def Model(samples_P_before_C, samples_P_after_W_C, samples_P_before_odtR1_C,samples_P_before_odtD1_C,IN, std, mean, F_out):
    N = samples_P_before_C.get_shape()[-1].value
    OUT = tf.reduce_sum(samples_P_before_C, axis=-1)  # (B,P,N)
    REM = IN - OUT
    # w = tf.Variable(tf.glorot_uniform_initializer()(shape=(N, 1)),
    #                 dtype=tf.float32, trainable=True, name='weight')
    # w2 = tf.Variable(tf.glorot_uniform_initializer()(shape=(N, 1)),
    #                 dtype=tf.float32, trainable=True, name='weight2')

    P_step = samples_P_before_C.get_shape()[1].value
    N = samples_P_before_C.get_shape()[-1].value
    # w 和 P的维度一样
    w = tf.Variable(tf.glorot_uniform_initializer()(shape=(P_step, N, N)),
                    dtype=tf.float32, trainable=True, name='weight')

    # w = tf.nn.sigmoid(w)
    P1 = model_1(samples_P_after_W_C)

    P2 = model_4(samples_P_before_odtR1_C,samples_P_before_odtD1_C)*w
    preds = tf.nn.softmax(P1  + P2)*tf.expand_dims(REM,axis=-1)

    # preds = tf.nn.softmax(P1) * tf.expand_dims(REM, axis=-1)

    outputs = preds + samples_P_before_C
    # outputs = model_common.multi_fc(outputs, activations="relu", units=[N], drop_rate=None, bn=False, dims=None,
    #                                is_training=True)
    In_preds = tf.reduce_sum(outputs, axis=-1)  # (M,P,N) #这个没用的,只是为了代码可以运行而已
    Out_preds = tf.reduce_sum(outputs, axis=-2)  # (M,P,N) #这个没用的,只是为了代码可以运行而已
    return [outputs,In_preds,Out_preds]



# def model_3(samples_P_before_C, samples_P_after_W_C, IN):
#     OUT = tf.reduce_sum(samples_P_before_C,axis=-1) #(B,P,N)
#     REM = IN - OUT
#     SUM = tf.expand_dims(tf.reduce_sum(samples_P_after_W_C,axis=-1),axis=-1) #(B,P,N)=>(B,P,N,1)
#     SUM = tf.where(tf.equal(SUM, 0.0), tf.ones_like(SUM), SUM)
#     # 如果最后一个维度的值是0，则用1代替，但如果标准化数据后，倒不必不会是0
#     P = samples_P_after_W_C / SUM #(B,P,N,N)
#     print(P.shape)
#     P_step = samples_P_before_C.get_shape()[1].value
#     N = samples_P_before_C.get_shape()[-1].value
#     # w 和 P的维度一样
#     w = tf.Variable(tf.glorot_uniform_initializer()(shape=(P_step, N, N)),
#                     dtype=tf.float32, trainable=True, name='weight')
#     # 检查是否有nan or inf
#     after_pred = tf.expand_dims(REM,axis=-1) * tf.nn.softmax(tf.add(P,w)) #(B,P,N,1)*(B,P,N,N)
#     pred = after_pred + samples_P_before_C
#     return pred
#
# def Model(samples_P_before_C, samples_P_after_W_C, samples_P_before_odt_C,IN, std, mean, F_out):
#     outputs = model_3(samples_P_before_C, samples_P_after_W_C, IN)
#     In_preds = tf.reduce_sum(outputs, axis=-1)  # (M,P,N) #这个没用的,只是为了代码可以运行而已
#     Out_preds = tf.reduce_sum(outputs, axis=-2)  # (M,P,N) #这个没用的,只是为了代码可以运行而已
#     return [outputs,In_preds,Out_preds]

# -*- coding: UTF-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import libs.model_common
# import model_TR_1 as TR

# x=(M,B,P,N,N+F) ,y=(M,B,N,N)
def placeholder(P,D,W, N, F_in, F_out):
    samples_P = tf.compat.v1.placeholder(shape = (None, P, N, F_in), dtype = tf.float32,name="samples_P")
    samples_D = tf.compat.v1.placeholder(shape = (None, D, N, F_in), dtype = tf.float32,name="samples_D")
    samples_W = tf.compat.v1.placeholder(shape = (None, W, N, F_in), dtype = tf.float32,name="samples_W")
    labels = tf.compat.v1.placeholder(shape = (None, N, F_out), dtype = tf.float32,name="lables")
    return labels, samples_P, samples_D,samples_W

def add_weight(X):
    w = tf.Variable(tf.glorot_uniform_initializer()(shape=()),
                    dtype=tf.float32, trainable=True, name='weight')  # (F1,)
    output = X*w
    return output

def Model(args, mean, std, Xs, SE, N,bn=False, bn_decay=None):
    T = args.intervals
    H = args.heads
    d = args.d
    F_out = N
    X_P = module(args, "P", mean, std, Xs[0], SE, T, H, d) #(B,N,N)
    X_D = module(args, "D", mean, std, Xs[1], SE, T, H, d) #(B,N,N)
    X_W = module(args,"W", mean, std, Xs[2], SE, T, H, d) #(B,N,N)

    # 方案一：太差
    # X = tf.concat([X_P,X_D,X_W],axis=-1) #(B,N,3N)

    # 方案三
    X_P = add_weight(X_P)
    X_D = add_weight(X_D)
    X_W = add_weight(X_W)

    # 方案二：比较好
    X = tf.add_n([X_P,X_D,X_W]) #(B,N,N)
    outputs = libs.model_common.multi_targets(X, std, mean, F_out)


    return outputs


def module(args, data_name, mean, std, X, SE, T, H, d, bn=False, bn_decay=None):
    D = H * d
    B,P,N,_ = X.get_shape().as_list()
    # (N,Fs)=>(1,1,N,D)
    SE = libs.model_common.s_embbing_static(SE, D, activations=['relu', None])
    # (B,P,N,2)=>(B,P,N,D)
    TE = libs.model_common.t_embbing_static(2, tf.cast(X[..., -2:], tf.int32), T, D, activations=['relu', None])
    # (B,P,N,N)
    X = libs.model_common.x_embedding(X[..., :-2], D, activations=['relu', None])

    if args.is_SE:
        # print("use SE")
        is_SE = True
    else:
        # print("no SE")
        is_SE=False
    if args.is_TE:
        # print("use TE")
        is_TE = True
    else:
        # print("no TE")
        is_TE = False
    # (B,P,N,D)=> (B,P,N,2D)
    # query_key_X = libs.model_common.x_SE_TE(X, SE, TE, is_X=True, is_SE=True, is_TE=True)
    query_key_X = libs.model_common.x_SE_TE(X, SE, TE, is_X=True, is_SE=is_SE, is_TE=is_TE)
    # query_key_X = libs.model_common.x_SE_TE(X, SE, TE, is_X=True, is_SE=False, is_TE=True) 20210526 01方案
    # query_key_X = libs.model_common.x_SE_TE(X, SE, TE, is_X=True, is_SE=True, is_TE=False) 20210526 02方案
    # (B,P,N,2D)=> (B,P,N,D)
    query = libs.model_common.multi_fc(query_key_X, activations=['relu'], units=[D])
    key = libs.model_common.multi_fc(query_key_X, activations=['relu'], units=[D])
    # (B,P,N,N) * (B,P,N,D)=> (B,P,N,D)
    value_X = libs.model_common.x_SE_TE(X, SE, TE, is_X=True, is_SE=is_SE, is_TE=is_TE)
    value = libs.model_common.multi_fc(value_X, activations=['relu'], units=[D])

    # (B,P,N,H*d)=>(BH,P,N,d)
    query = tf.concat(tf.split(query, H, axis = -1), axis = 0)
    key = tf.concat(tf.split(key, H, axis = -1), axis = 0)
    value = tf.concat(tf.split(value, H, axis = -1), axis = 0)

    # (BH,P,N,d)*(BH,P,N,d)=>(BH,P,N,N)
    attention = tf.matmul(query, key, transpose_b = True)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis = -1)
    # (BH,P,N,N) * (BH,P,N,d)=> (BH,P,N,d)
    X = tf.matmul(attention, value)
    # (BH,P,N,d) => (B,P,N,Hd)
    X = tf.concat(tf.split(X, H, axis = 0), axis = -1)

    X = libs.model_common.multi_fc(X, activations=['relu',None], units=[D,N]) # (B,P,N,N)
    if args.is_RNN:
        # print("use RNN")
        X = RNNs_end(args, X, P, N, data_name)# 202010526 去掉RNN
    else:
        # print("use FC")
        X = FC_end(args,X,P,N)
    # 方案一
    # X = libs.model_common.inverse_positive(X, std, mean, N)
    # eixt()
    return X

def FC_end(args, X,P,N):
    # (B,P,N,N)=>(B,N,PN)
    X = tf.reshape(X,[-1,N,P*N]) # fc层
    X = libs.model_common.multi_fc(X, activations=['relu'], units=args.ANN_units) # (B,P,N,N)
    return X

def RNNs_end(args, X, P, N, data_name):
    with tf.variable_scope(data_name, reuse=tf.AUTO_REUSE):
        X = tf.reshape(tf.transpose(X,[0,2,1,3]),[-1,P,N])#(B,N,P,N)=>(BN,P,N)
        X = libs.model_common.multi_lstm(X, args.RNN_units, args.RNN_type) #(BN,P,N)=>(BN,N)
        X = tf.reshape(X, [-1,N,X.shape[-1]]) #(BN,N) =>(B,N,N)
    return X

def TCN_end(args, X):
    x = tf.transpose(X,[0,2,1,3]) #(B,N,P,N)
    y =  libs.model_common.multi_tcn(x, args.K, args.d, activations=args.TCN_activations, units=args.TCN_units)
    return y[:,:,-1,:] #(b,n,n)
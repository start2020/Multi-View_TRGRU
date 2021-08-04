# -*- coding: UTF-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from libs import model_common

# x=(M,B,P,N,N+F) ,y=(M,B,N,N)
def placeholder(P, N, F_in, F_out):
    samples = tf.compat.v1.placeholder(shape = (None, P, N, F_in), dtype = tf.float32)
    labels = tf.compat.v1.placeholder(shape = (None, N, F_out), dtype = tf.float32)
    #is_training = tf.compat.v1.placeholder(shape = (), dtype = tf.bool)
    return labels, samples

def get_attention(X, D):
    # (B,P,N,2D)=> (B,P,N,D)
    query = model_common.multi_fc(X, activations=['relu'], units=[D])
    key = model_common.multi_fc(X, activations=['relu'], units=[D])
    # (B,P,N,D)*(B,P,N,D)=>(B,P,N,N)
    attention = tf.matmul(query, key, transpose_b = True)
    attention /= (D ** 0.5)
    attention = tf.nn.softmax(attention, axis = -1)
    return attention


def Model(args, mean, std, X, SE, bn=False, bn_decay=None):
    T = args.intervals
    H = args.heads
    D = args.d
    B,P,N,_ = X.get_shape().as_list()
    # (N,Fs)=>(1,1,N,D)
    SE = model_common.s_embbing_static(SE, D, activations=['relu', None])
    #(B,P,N,2)=>(B,P,N,D)
    TE = model_common.t_embbing_static(2, tf.cast(X[...,-2:],tf.int32), T, D, activations=['relu', None])
    # (B,P,N,N)
    X = X[..., :-2]
    X = model_common.x_embedding(X, D, activations=['relu', None])

    # (B,P,N,D)=>(B,P,N,N)
    query_key = model_common.x_SE_TE(X, SE, TE, is_X=False, is_SE=True, is_TE=True)
    attention = get_attention(query_key, D)
    # (B,P,N,N) * (B,P,N,D)=> (B,P,N,D)

    value = model_common.multi_fc(X, activations=['relu'], units=[D])
    X = tf.matmul(attention, value)

    X = model_common.multi_fc(X, activations=['relu',None], units=[D,N]) # (B,P,N,N)
    X = RNNs_end(args, X, P, N)
    #X = FC_end(args,X,P,N)
    F_out = N
    outputs = model_common.multi_targets(X, std, mean, F_out)
    return outputs


def FC_end(args, X,P,N):
    # (B,P,N,N)=>(B,N,PN)
    X = tf.reshape(X,[-1,N,P*N]) # fc层
    X = model_common.multi_fc(X, activations=['relu'], units=args.ANN_units) # (B,P,N,N)
    return X

def RNNs_end(args, X,P,N):
    X = tf.reshape(tf.transpose(X,[0,2,1,3]),[-1,P,N])#(B,N,P,N)=>(BN,P,N)
    X = model_common.multi_lstm(X, args.RNN_units, args.RNN_type) #(BN,P,N)=>(BN,N)
    X = tf.reshape(X, [-1,N,X.shape[-1]]) #(BN,N) =>(B,N,N)
    return X

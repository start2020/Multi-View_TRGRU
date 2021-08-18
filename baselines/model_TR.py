# -*- coding: UTF-8 -*-
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import libs.model_common

# x=(M,B,P,N,N+F) ,y=(M,B,N,N)
def placeholder(P, N, F_in, F_out):
    samples = tf.compat.v1.placeholder(shape = (None, P, N, F_in), dtype = tf.float32,name="samples")
    labels = tf.compat.v1.placeholder(shape = (None, N, F_out), dtype = tf.float32,name="lables")
    #is_training = tf.compat.v1.placeholder(shape = (), dtype = tf.bool)
    return labels, samples

def Model(args, mean, std, X, SE, bn=False, bn_decay=None):
    T = args.intervals
    H = args.heads
    d = args.d
    D = H * d
    B,P,N,_ = X.get_shape().as_list()
    SE = libs.model_common.s_embbing_static(SE, D, activations=['relu', None]) #(N,Fs)=>(1,1,N,D)
    TE = tf.cast(X[...,-2:],tf.int32) #(B,P,N,2)
    TE = libs.model_common.t_embbing_static(2, TE, T, D, activations=['relu', None]) #(B,P,N,D)
    X = X[..., :-2]  # (B,P,N,N)
    X = libs.model_common.x_embedding(X, D, activations=['relu', None])
    X = libs.model_common.x_SE_TE(X, SE, TE, is_X=True, is_SE=True, is_TE=True)
    #X = libs.model_common.x_spatio_temporal(X, SE, TE, activations=['relu', None])  # (B,P,N,2D)
    # (B,P,N,2D)=> (B,P,N,D)
    query = libs.model_common.multi_fc(X, activations=['relu'], units=[D])
    key = libs.model_common.multi_fc(X, activations=['relu'], units=[D])
    value = libs.model_common.multi_fc(X, activations=['relu'], units=[D])

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
    X = RNNs_end(args, X, P, N)
    #X = FC_end(args,X,P,N)
    F_out = N
    outputs = libs.model_common.multi_targets(X, std, mean, F_out)
    return outputs


def FC_end(args, X,P,N):
    # (B,P,N,N)=>(B,N,PN)
    X = tf.reshape(X,[-1,N,P*N]) # fc层
    X = libs.model_common.multi_fc(X, activations=['relu'], units=args.ANN_units) # (B,P,N,N)
    return X

def RNNs_end(args, X,P,N):
    X = tf.reshape(tf.transpose(X,[0,2,1,3]),[-1,P,N])#(B,N,P,N)=>(BN,P,N)
    RNN_units_list = [args.RNN_units]
    X = libs.model_common.multi_lstm(X, RNN_units_list, args.RNN_type)  # (BN,P,N)=>(BN,N)
    # X = libs.model_common.multi_lstm(X, args.RNN_units, args.RNN_type) #(BN,P,N)=>(BN,N)
    X = tf.reshape(X, [-1,N,X.shape[-1]]) #(BN,N) =>(B,N,N)
    return X

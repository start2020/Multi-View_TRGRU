import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import libs.utils, libs.metrics, libs.main_common, libs.data_common, libs.model_common
import numpy as np

# samples=(M,B,T,N,F_in), labels=(M,B,N,F_out)
def placeholder(T, N, F_in, F_out):
    graph_sem = tf.compat.v1.placeholder(shape = (None, T, N, N), dtype = tf.float32,name="graph_sem")
    samples = tf.compat.v1.placeholder(shape = (None, T, N, F_in), dtype = tf.float32,name="samples")
    labels = tf.compat.v1.placeholder(shape = (None, N, F_out), dtype = tf.float32,name="labels")
    return [labels, samples, graph_sem]

def Model(args, mean, std, samples, graph_geo, graph_sem):
    # print("graph_geo ",graph_geo)
    print("graph_sem ", graph_sem)
    print("samples ",samples)
    N = graph_geo.shape[-1]
    R_geo = libs.model_common.multi_gcn(graph_geo, samples, activations=args.GCN_activations, units=args.GCN_units, Ks=args.Ks) #(B,T,N,128)
    print("R_geo ",R_geo)
    S_geo = libs.model_common.multi_gcn(graph_sem, samples, activations=args.GCN_activations, units=args.GCN_units, Ks=args.Ks) #(B,T,N,128)
    print("S_geo ",S_geo)
    V = tf.concat([R_geo, S_geo], axis=-1) #(B,T,N,256)
    V = tf.transpose(V,perm=[0,2,1,3]) #(B,N,T,256)
    V = tf.reshape(V, shape=(-1, V.shape[-2], V.shape[-1])) #(BN,T,256)
    print("V ",V)
    # (BN,T,256) => H,shape=(BN,T,128) => shape=(BN,128)
    H = libs.model_common.multi_lstm(V, args.RNN_units, args.RNN_type)
    print("H ",H)
    H = tf.reshape(H,shape=[-1, N, H.shape[-1]]) #(BN,128) =>(B,N,128)
    H_m = libs.model_common.fc_layer(H, units=128) # (B,N,128)=>(B,N,128)
    M_pred = tf.matmul(H_m, H, transpose_b=True) #(B,N,128)*(B,N,128)=>(B,N,N)
    print("M_pred ",M_pred)
    P_pred = libs.model_common.fc_layer(H, units=1) #(B,N,128)=>(B,N,1)
    Q_pred = libs.model_common.fc_layer(H,units=1) #(B,N,128)=>(B,N,1)
    M_pred = libs.model_common.inverse_positive(M_pred, std, mean, M_pred.shape[-1])
    print("M_pred ",M_pred)
    P_pred = libs.model_common.inverse_positive(P_pred, std, mean, P_pred.shape[-1])
    Q_pred = libs.model_common.inverse_positive(Q_pred, std, mean, Q_pred.shape[-1])
    exit()
    return [M_pred,P_pred,Q_pred]
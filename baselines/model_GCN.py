import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import libs.model_common


# X=(M,B,N,PN) ,y=(M,B,N,N)
def placeholder(N, F_in, F_out):
    samples = tf.compat.v1.placeholder(shape = (None, N, F_in), dtype = tf.float32,name="samples")
    labels = tf.compat.v1.placeholder(shape = (None, N, F_out), dtype = tf.float32,name="lables")
    return labels, samples

def Model(args, mean, std, X, graph, F_out):
    X = libs.model_common.multi_gcn(graph, X, activations=args.GCN_activations, units=args.GCN_units, Ks=args.Ks) #(B,T,N,128)
    outputs = libs.model_common.multi_targets(X, std, mean, F_out)
    return outputs
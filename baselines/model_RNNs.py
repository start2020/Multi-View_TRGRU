import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import libs.model_common

# X:(M,T,N,N)=>(M*N,T,N), Y:(M,N,N)=>(M*N,N)
def placeholder(T, F_in, F_out):
    samples = tf.compat.v1.placeholder(shape = (None,T, F_in), dtype = tf.float32,name="samples")
    labels = tf.compat.v1.placeholder(shape = (None, F_out), dtype = tf.float32,name="lables")
    return labels, samples

# X=(B,T,F)
def Model(args, mean, std, X, F_out):
    output = libs.model_common.multi_lstm(X, args.units, type=args.RNN_Type) #(B,F)
    # output = libs.model_common.multi_fc(output)
    outputs = libs.model_common.multi_targets(output, std, mean, F_out)
    return outputs


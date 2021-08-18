import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import libs.model_common

'''
预测目标可以是(M,B,N,N), 也可以是(M,B,N,N,1)
'''
# X=(M,B,N,PN) ,y=(M,B,N,N)
def placeholder_vector(N, F_in, F_out):
    samples = tf.compat.v1.placeholder(shape = (None, N, F_in), dtype = tf.float32,name="samples")
    labels = tf.compat.v1.placeholder(shape = (None, N, F_out), dtype = tf.float32,name="lables")
    return labels, samples

# X=(M,B,N,N,P) ,y=(M,B,N,N,1)
def placeholder_scalar(N, F_in, F_out):
    samples = tf.compat.v1.placeholder(shape = (None, N, N, F_in), dtype = tf.float32,name="samples")
    labels = tf.compat.v1.placeholder(shape = (None, N, N, F_out), dtype = tf.float32,name="lables")
    return samples, labels

def placeholder_training():
    is_training = tf.compat.v1.placeholder(shape=(),dtype=tf.bool, name="is_training")
    return is_training

def Model(args, mean, std, X, F_out,drop_rate=None, bn=False, dims=None, is_training=True):
    X = libs.model_common.multi_fc(X, activations=args.activations, units=args.units, drop_rate=drop_rate, bn=bn, dims=dims, is_training=is_training)
    outputs = libs.model_common.multi_targets(X, std, mean, F_out)
    return outputs
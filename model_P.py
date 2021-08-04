import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from libs import model_common

'''
输入：(M,P,N,N) before
输出：(M,P,N,N) after; (M,P,N,N)all
'''
# X=(M,B,T,N,F_in)
def placeholder(T, N, F_in,F_out):
    samples = tf.compat.v1.placeholder(shape = (None, T, N, F_in), dtype = tf.float32)
    labels = tf.compat.v1.placeholder(shape=(None,T,N,F_out), dtype=tf.float32)
    return labels, samples

def Model(args, mean, std, X, F_out,drop_rate=None, bn=False, dims=None, is_training=True):
    X = model_common.multi_fc(X, activations=args.activations, units=args.units, drop_rate=drop_rate, bn=bn, dims=dims, is_training=is_training)
    outputs = model_common.multi_targets(X, std, mean, F_out)
    return outputs

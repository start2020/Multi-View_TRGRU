import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from libs import model_common

'''
输入：(M,P,N,N) before
输出：(M,P,N,N) after; (M,P,N,N)all
'''
# X=(M,B,T,N,F_in)
def placeholder(P, N):
    samples = tf.compat.v1.placeholder(shape = (None, N, N), dtype = tf.float32)
    labels = tf.compat.v1.placeholder(shape=(None,N, N), dtype=tf.float32)
    return labels, samples

def Model(args, mean, std, X, F_out,drop_rate=None, bn=False, dims=None, is_training=True):

    W = tf.Variable(tf.glorot_uniform_initializer()(shape=[X.shape[-1], units]),
                    dtype=tf.float32, trainable=True, name='kernel')  # (F, F1)
    b = tf.Variable(tf.zeros_initializer()(shape=[units]),
                    dtype=tf.float32, trainable=True, name='bias')  # (F1,)
    Y = tf.matmul(X, W) + b  # (...,F)*(F,F1)+(F1,)=>(...,F1)

    for i in range(3):
        tmp = tf.reduce_sum(X,-1)
        x_i = X[:,i,:,:]/tmp*Wi+

    return outputs

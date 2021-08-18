import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import libs.model_common

# B,N,P = 10, 20, 5
# inflow = tf.random.normal(shape=(B,N,1,P))
# outflow = tf.random.normal(shape=(B,N,1,P))
# X = tf.random.normal(shape=(B,N,N,P))
# output_dims= [2,1]
# kernel_size1,kernel_size2 = [3,3],[5,5]
# gpu_options = tf.GPUOptions(allow_growth=True)
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#     sess.run(tf.global_variables_initializer())
#     output = sess.run([output])
#     print(output[0].shape)
# X=(M,B,N,PN) ,y=(M,B,N,N)

def placeholder_vector(P,N):
    labels = tf.compat.v1.placeholder(shape=(None, N, N), dtype=tf.float32, name="lables")
    samples = tf.compat.v1.placeholder(shape=(None, N, N, P), dtype=tf.float32, name="samples")
    in_flow = tf.compat.v1.placeholder(shape = (None, N,1,P), dtype = tf.float32,name="in_flow")
    out_flow = tf.compat.v1.placeholder(shape = (None, N, 1, P), dtype = tf.float32,name="out_flow")


    return labels, samples, in_flow, out_flow

#output_dims=[2.1],kernel_size1=[3,3], kernel_size2=[5,5]
def Model(args, std, mean, X, inflow, outflow):
    output_dims, kernel_size1, kernel_size2 = args.output_dims, args.kernel_size1, args.kernel_size2
    trunk_output = multi_layer(X, output_dims, kernel_size1, kernel_size2)  # (B,N,N,1)
    o_inout = inout_gate(inflow, outflow)  # (B,N,1,1)
    output = trunk_output + o_inout  # (B,N,N,1)+(B,N,1,1)=>(B,N,N,1)
    output = libs.model_common.conv2d_layer(output, 1, [1, 1], stride=[1, 1], padding='SAME')
    output = tf.squeeze(output,axis=-1)
    F_out = output.shape[-1] #(B,N,N)
    outputs = libs.model_common.multi_targets(output, std, mean, F_out)
    return outputs

# (B,N,N,C) => (B,1,1,C)
def channel_wise(X):
    Pool_Size = [X.shape[1], X.shape[2]] #(N,N)
    X = tf.layers.average_pooling2d(X, pool_size=Pool_Size, strides=[1,1])
    C = X.shape[3]
    # (B,1,1,1) =>(B,1,1,C)
    X = libs.model_common.multi_fc(X, activations=['relu','sigmoid'], units=[1, C])
    return X

# (B,N,N,C)=>(B,N,N,F)
def block(X, output_dims, kernel_size):
    # (B,N,N,C)=>(k,k),F=>(B,N,N,F)
    # (10,20,20,1)
    X = libs.model_common.conv2d_layer(X, output_dims, kernel_size, stride=[1, 1],padding='SAME')
    #(B,N,N,F) => (B,1,1,F)
    E = channel_wise(X)
    # (B,N,N,F) * (B,1,1,F) =>(B,N,N,F)
    Y = X*E
    return Y

# (B,N,N,C)=>(B,N,N,F)
def layer(X, output_dim, kernel_size1, kernel_size2):
    Y1 = block(X, output_dim, kernel_size1) # (B,N,N,C)=>(B,N,N,F)
    Y2 = block(X, output_dim, kernel_size2) # (B,N,N,C)=>(B,N,N,F)
    output = Y1 + Y2  # (B,N,N,F)
    return output

# (B,N,N,C)=>(B,N,N,1), output_dims=[2,1]
def multi_layer(X, output_dims, kernel_size1, kernel_size2):
    for output_dim in output_dims:
        output = layer(X, output_dim, kernel_size1, kernel_size2) #(B,N,N,C)=>(B,N,N,F)
        X = output
    return output

# (B,N,1,P) =>(B,N,1,1)
def inout_gate(inflow, outflow):
    # (B,N,1,P)=>(B,N,1,1)
    o_in = libs.model_common.conv2d_layer(inflow, 1, [1,1], stride=[1, 1],padding='SAME')
    o_out = libs.model_common.conv2d_layer(outflow, 1, [1,1], stride=[1, 1],padding='SAME')
    N = o_out.get_shape().as_list()[1]
    w = tf.Variable(tf.glorot_uniform_initializer()(shape=[N,1,1]), dtype=tf.float32, trainable=True, name='kernel')
    #(B,N,1,1)*(N,1,1)=>(B,N,1,1)
    o_inout = o_in * o_out * w
    return o_inout






import numpy as np
import pandas as pd

# 检查 入闸分布和入闸总人数之间的区别
in_matrix = np.load("./sz/matrix_in_15_07.npz")["in_matrix"]
print("in_matrix.shape: ", in_matrix.shape)
# (D,T1,N,N,T2)
D,T1,N,N,T2 = in_matrix.shape
x1 = np.reshape(in_matrix, [D*T1,N,N*T2])
x2 = np.sum(x1,axis=-1)
x2 = x2[:1920,...]
print(x2.shape)

in_all = pd.read_excel("./sz/all_in_15_07.xlsx",index_col=0)
print("all_in.shape", in_all.values.shape)
x3 = in_all.values - x2
df = pd.DataFrame(x3, index=in_all.index)
df.to_excel("./sz/diff.xlsx")

from libs import utils, data_common,para
import os
import numpy as np
def main(args):
    print(args)
    data_mode = args.data_mode
    dir_name, data_path, log = utils.path(args,data_mode) # 创建存放数据和日志的文件夹
    M = data_common.load_original_data(args, log)  # 原始数据格式是NPZ,key是"matrix"
    dayofweek = data_common.week_transform(args, log) #时间特征，(Days, T1, N, T)
    M = np.sum(M,axis=-1)
    Days, T1, N, _ = M.shape
    M = np.reshape(M,(Days,T1,N*N)) #(Days,T1,N*N)
    Dayoftime = np.tile(np.reshape(np.array([t for t in range(T1)]), (1, T1, 1)), (Days, 1, 1))  # (Days,T1,1)
    Dayofweek = np.tile(np.reshape(np.array(dayofweek), (Days, 1, 1)), (1, T1, 1))  # (Days,T1,1)
    Dayofyear = np.tile(np.reshape(np.arange(len(dayofweek)), (Days, 1, 1)), (1, T1, 1))  # (Days,T1,1)
    output = np.concatenate([Dayoftime, Dayofweek, Dayofyear], axis=-1)  # (Days,T1,3)
    M = np.concatenate((M,output),axis=-1)
    M = np.reshape(M,(Days*T1,N*N+3))#(Days*T1,N*N+3)
    M_len =(Days*T1)

    previous = args.previous
    # previous = 32 * 7
    prediction = 1

    # M = M[:numbers*previous]
    all_data = [[], []]
    # numbers = M_len // previous
    # for i in range(numbers):
    #     if (i + 1) * previous+prediction>Days*T1: continue
    #     all_data[0].append(M[i*previous:(i+1)*previous])
    #     all_data[1].append(M[(i+1)*previous:(i + 1) * previous+prediction])

    for i in range(M_len):
        if i>80: continue
        if i + previous + prediction > M_len: continue
        all_data[0].append(M[i:i+previous])
        all_data[1].append(M[i+previous:i + previous + prediction])


    all_data[0] = all_data[0][0:60]
    all_data[1] = all_data[1][0:60]

    data_names = ['samples_P','labels']
    all_data = data_common.process_all(args, all_data, data_names,data_mode)

    data_common.save_feature_data(args, all_data, data_names,data_mode)



if __name__ == "__main__":
        parser = para.original_data_Mat_para()
        args = parser.parse_args()
        main(args)
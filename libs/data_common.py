import numpy as np
import os
import datetime
from libs import utils, graph_common
import pickle
'''
函数说明：分三类函数
一、原始数据的预处理
1-1 
二、所有模型共享的数据处理
2-1 
三、模型独自所需的数据处理
3-1 
'''
######################################## Data process for original ########################################
'''
产生随机数据
'''
def generate_data_random_TR(args):
    train_data,val_data, test_data = {},{},{}
    Q, M, B, P, N, F =1, 247, 64, args.P, 118, 118
    x = np.random.normal(size=(M,B,P,N,F)).astype(np.float32)
    y = np.random.normal(size=(M,B,N,N)).astype(np.float32)
    TE = np.random.normal(size=(M,B,P+Q,2)).astype(np.int32)
    train_data['x'], train_data['y'], train_data['TE'] = x, y, TE
    val_data['x'], val_data['y'], val_data['TE'] = x, y, TE
    test_data['x'], test_data['y'], test_data['TE'] = x, y, TE
    return train_data, val_data, test_data

def load_original_data(args, log):
    data_file = os.path.join(args.dataset_dir, 'original', args.data_file)
    M = np.load(data_file)["matrix"]
    M = M.astype(np.float32)
    # Days, T1, N, _, T2 = 30, 20, 10, 10, 20
    # M = np.random.normal(size=(Days, T1, N, _, T2)).astype(np.float32)
    utils.log_string(log, 'original data shape: ' + str(M.shape) + ' dtype: ' + str(
        M.dtype))
    return M

'''
产生周特征
'''
def week_transform(args,log):
    station_txt_file = os.path.join(args.dataset_dir, 'original', args.day_index)
    day_list = []
    with open(station_txt_file, "r") as fin:
        for line in fin.readlines():
            line = line[:-1]  # 去掉\n
            day_list.append(line)
    dayofweek = np.array([int(datetime.datetime.strptime(day_list[i], '%Y-%m-%d').strftime("%w")) for i in
                 range(len(day_list))])
    utils.log_string(log, 'week features shape: ' + str(dayofweek.shape) + ' dtype: ' + str(
        dayofweek.dtype))
    return dayofweek

'''
dayofweek: list
'''
def add_time(T1, N, dayofweek):
    Days = len(dayofweek)
    Dayoftime = np.tile(np.reshape(np.array([t for t in range(T1)]), (1, T1, 1, 1)),(Days, 1, N, 1)) #(Days,T1,N,1)
    Dayofweek = np.tile(np.reshape(np.array(dayofweek), (Days, 1, 1, 1)),(1, T1, N, 1)) # (Days,T1,N,1)
    Dayofyear = np.tile(np.reshape(np.arange(len(dayofweek)), (Days, 1, 1, 1)),(1, T1, N, 1)) # (Days,T1,N,1)
    output = np.concatenate([Dayoftime, Dayofweek, Dayofyear], axis=-1) # (Days,T1,N,3)
    return output

'''
列表聚合
'''
def process_all(args, all_data, data_names,data_mode):
    dir_name, data_path, log = utils.path(args,data_mode)
    for i in range(len(all_data)):
        # print(i,data_names[i])
        all_data[i] = np.stack(all_data[i], axis=0).astype(np.float32)

        message = '%s_%s_%s shape: %s dtype: %s' % (data_names[i], args.data_type, dir_name, str(all_data[i].shape), str(all_data[i].dtype))
        utils.log_string(log, message)
    return all_data

'''
功能：保存指定输入模式的数据，跟模型无关
data: 被保存的数据列表
data_names: 数据名称
data_path: 保存路径
'''
def save_feature_data(args, all_data, data_names,data_mode):
    data_path = os.path.join(args.dataset_dir, args.model, data_mode, utils.input_name(args)) # 路径
    print("data_path:",data_path)
    for i in range(len(data_names)):
        save_name = os.path.join(data_path, data_names[i] + '_' + args.data_type + '.npz')
        if all_data[i].shape[1]!=0:
            np.savez_compressed(save_name, all_data[i])

'''
功能：装载指定的数据
返回：一个列表，元素是x,x1,..,y
'''
def load(args, log, dir_name):
    path = os.path.join(args.dataset_dir, 'all_models', args.data_mode, dir_name) # 存放的文件夹路径
    results = []
    data_names_list = args.data_names.split("-")
    for data_name in data_names_list:
        file_name = data_name + '_' + args.data_type + '.npz'
        file_path = os.path.join(path, file_name)
        result = np.load(file_path)['arr_0']
        utils.log_string(log, '%s %s' % (file_name, str(result.shape)))
        results.append(result)
    return results

# Time = add_time(T1,N,dayofweek)  # (D,T1,N,T)
def sequence_OD_data_PDW2(args, M_OD, Time, dow_num=7):
    # 前P个时间段,前D天,前W周
    P = int(args.input_steps[0])
    D = int(args.input_steps[1])
    W = int(args.input_steps[2])

    M = np.sum(M_OD, axis=-1)  # (D,T1,N,N)

    Days, T1, N, _ = M.shape

    if args.data_type == "In" or args.data_type == "Out":
        M = np.expand_dims(M, axis=-1)  # (Days, T1, N, 1)
    M = np.concatenate([M, Time], axis=-1)  # (Days, T1, N, F+T)

    all_data = [[], [], [], [], [], []]
    for j in range(Days):
        if j - dow_num * W < 0: continue
        weeks = [j - dow_num * w for w in range(1, W + 1)]  # [j-7,...,j-7W]
        if j - D < 0: continue
        for i in range(T1):
            if i - P < 0: continue
            y = M[j, i, ...]  # 第j天第i个时间段，(N,F)
            x_D = M[j - D:j, i, ...]  # 前D天第i个时间段，(D,N,F)
            x_W = M[weeks, i, ...]  # 前W周同一天第i个时间段(W,N,F)
            x_P = M[j, i - P:i, ...]  # 同一天，前P个时间段，全天候出闸，(P,N,F)
            all_data[0].append(x_P)  # (M,P,N,F)
            all_data[1].append(x_D)  # (M,D,N,F)
            all_data[2].append(x_W)  # (M,W,N,F)
            all_data[3].append(y)  # (M,N,F)

            x_P_before = np.sum(M_OD[j, i - P:i, ..., :i], axis=-1)  # 同一天，前P个时间段，预测时间段前出闸，(P,N,N)
            x_P_after = np.sum(M_OD[j, i - P:i, ..., i:], axis=-1)  # 同一天，前P个时间段，预测时间段后出闸，(P,N,N)
            time = Time[j, i - P:i, ...]  # (P,N,3)
            x_P_before = np.concatenate([x_P_before, time], axis=-1)  # (P,N,N+3)
            x_P_after = np.concatenate([x_P_after, time], axis=-1)  # (P,N,N+3)
            all_data[4].append(x_P_before)
            all_data[5].append(x_P_after)
    return all_data


def sequence_OD_data_all_C(args, M_OD, dayofweek,data_mode):
    # 前P个时间段,前D天,前W周
    P = int(args.input_steps[0])
    D = int(args.input_steps[1])
    W = int(args.input_steps[2])
    Days, T1, N, N, T2 = M_OD.shape
    M = np.sum(M_OD, axis=-1)  # (D,T1,N,N)
    M_OD_t = np.sum(M_OD, axis=1) #(D,N,N,T2)

    M1= M
    if args.data_type=="In" or args.data_type=="Out":
        M = np.expand_dims(np.sum(M_OD,axis=-1), axis=-1)  # (Days, T1, N, 1)
    Time = add_time(T1,N,dayofweek)  # (D,T1,N,T)
    M = np.concatenate([M, Time], axis=-1)  # (Days, T1, N, F+T)
    all_data = [[] for x in range(7+1)]
    for j in range(Days):
        # print(j)
        if j - 7 * W -7 < 0: continue
        weeks = [j - 7 * w for w in range(1, W+1)] #[j-7,...,j-7W]
        if j - D -7 < 0: continue
        for i in range(T1):
            complete_P = 3
            if i - (P+complete_P) < 0: continue
            y = M[j, i, ...]  # 第j天第i个时间段，(N,F)
            x_D = M[j-D:j,i,...] # 前D天第i个时间段，(D,N,F)
            x_W = M[weeks,i,...] # 前W周同一天第i个时间段(W,N,F)
            x_P = M[j,i-P:i, ...]  # 同一天，前P个时间段，全天候出闸，(P,N,F)
            In_P = np.sum(M1[j, i-P:i, ...], axis=-1)  # 同一天，前P个时间段，(P,N,)
            all_data[0].append(x_P)  # (M,P,N,F)
            all_data[1].append(x_D)  # (M,D,N,F)
            all_data[2].append(x_W)  # (M,W,N,F)
            all_data[3].append(y)  # (M,N,F)
            all_data[4].append(In_P) #(M,P,N,)

            x_P_before = np.sum(M_OD[j, i - P:i, ..., :i], axis=-1)  # 同一天，前P个时间段，预测时间段前出闸，(P,N,N)
            # 最后一维添加三个没用的维度,在数据处理的时候会去掉
            tmp_mat = np.expand_dims(np.zeros(x_P_before.shape[:-1],dtype=np.float32),axis=-1)
            x_P_before = np.concatenate((x_P_before,tmp_mat),axis=-1)
            x_P_before = np.concatenate((x_P_before, tmp_mat), axis=-1)
            x_P_before = np.concatenate((x_P_before, tmp_mat), axis=-1)

            odt_R123 = []
            for k in range(P):
                # od = M_OD_t[k, ..., i - P + k - 3:i - P + k]  # 同一天，前P个时间段，预测时间段前出闸，(N,N,3)
                od = M_OD_t[j, ..., i - P + k - 3:i - P + k]  # 同一天，前P个时间段，预测时间段前出闸，(N,N,3)
                # od = M_OD_t[..., i - P + k - 3:i - P + k]  # 同一天，前P个时间段，预测时间段前出闸，(N,N,3)
                odt_R123.append(od)
            odt_R123 = np.stack(odt_R123, axis=0)  # (P,N,N,3)

            x_P_after_W = np.sum(M_OD[j - 7, i - P:i, ..., i:], axis=-1)  # 前7天，前P个时间段，预测时间段后出闸，(P,N,N)


            all_data[5].append(x_P_before)
            all_data[6].append(x_P_after_W)
            all_data[7].append(odt_R123)



    data_names_1 = ['samples_P_C','samples_D_C', 'samples_W_C', 'labels_C','In_C']
    data_names_2 = ['samples_P_before_C', 'samples_P_after_W_C', 'samples_P_before_odt123_C']
    data_names = data_names_1 + data_names_2
    all_data = process_all(args, all_data, data_names,data_mode)

    save_feature_data(args, all_data, data_names,data_mode)
    return all_data, data_names

def sequence_OD_data_all_C_V2(args, M_OD, dayofweek,data_mode):
    # 前P个时间段,前D天,前W周
    P = int(args.input_steps[0])
    D = int(args.input_steps[1])
    W = int(args.input_steps[2])
    Days, T1, N, N, T2 = M_OD.shape
    M = np.sum(M_OD, axis=-1)  # (D,T1,N,N)
    M_OD_t = np.sum(M_OD, axis=1) #(D,N,N,T2)

    M1= M
    if args.data_type=="In" or args.data_type=="Out":
        M = np.expand_dims(np.sum(M_OD,axis=-1), axis=-1)  # (Days, T1, N, 1)
    Time = add_time(T1,N,dayofweek)  # (D,T1,N,T)
    M = np.concatenate([M, Time], axis=-1)  # (Days, T1, N, F+T)
    all_data = [[] for x in range(8+1)]
    for j in range(Days):
        # print(j)
        if j - 7 * W -7 < 0: continue
        weeks = [j - 7 * w for w in range(1, W+1)] #[j-7,...,j-7W]
        if j - D -7 < 0: continue
        for i in range(T1):
            complete_P = 3
            if i - (P+complete_P) < 0: continue
            y = M[j, i, ...]  # 第j天第i个时间段，(N,F)
            x_D = M[j-D:j,i,...] # 前D天第i个时间段，(D,N,F)
            x_W = M[weeks,i,...] # 前W周同一天第i个时间段(W,N,F)
            x_P = M[j,i-P:i, ...]  # 同一天，前P个时间段，全天候出闸，(P,N,F)
            In_P = np.sum(M1[j, i-P:i, ...], axis=-1)  # 同一天，前P个时间段，(P,N,)
            all_data[0].append(x_P)  # (M,P,N,F)
            all_data[1].append(x_D)  # (M,D,N,F)
            all_data[2].append(x_W)  # (M,W,N,F)
            all_data[3].append(y)  # (M,N,F)
            all_data[4].append(In_P) #(M,P,N,)

            x_P_before = np.sum(M_OD[j, i - P:i, ..., :i], axis=-1)  # 同一天，前P个时间段，预测时间段前出闸，(P,N,N)
            # 最后一维添加三个没用的维度,在数据处理的时候会去掉
            tmp_mat = np.expand_dims(np.zeros(x_P_before.shape[:-1],dtype=np.float32),axis=-1)
            x_P_before = np.concatenate((x_P_before,tmp_mat),axis=-1)
            x_P_before = np.concatenate((x_P_before, tmp_mat), axis=-1)
            x_P_before = np.concatenate((x_P_before, tmp_mat), axis=-1)

            odt_R123 = []
            for k in range(P):
                # od = M_OD_t[k, ..., i - P + k - 3:i - P + k]  # 同一天，前P个时间段，预测时间段前出闸，(N,N,3)
                od = M_OD_t[j, ..., i - P + k - 3:i - P + k]  # 同一天，前P个时间段，预测时间段前出闸，(N,N,3)
                # od = M_OD_t[..., i - P + k - 3:i - P + k]  # 同一天，前P个时间段，预测时间段前出闸，(N,N,3)
                odt_R123.append(od)
            odt_R123 = np.stack(odt_R123, axis=0)  # (P,N,N,3)

            odt_W123 = []
            for k in range(P):
                # od = M_OD_t[k, ..., i - P + k - 3:i - P + k]  # 同一天，前P个时间段，预测时间段前出闸，(N,N,3)
                od = M_OD_t[j-7, ..., i - P + k - 3:i - P + k]  # 同一天，前P个时间段，预测时间段前出闸，(N,N,3)
                # od = M_OD_t[..., i - P + k - 3:i - P + k]  # 同一天，前P个时间段，预测时间段前出闸，(N,N,3)
                odt_W123.append(od)
            odt_W123 = np.stack(odt_W123, axis=0)  # (P,N,N,3)

            # odt_R1 = M_OD_t[j,..., i - 1:i ]# (N,N,1)# 同一天，前1个时间段，预测时间段前出闸，(N,N,3)
            # odt_R1 = np.expand_dims(odt_R1,axis=0)# (1,N,N,1)
            # odt_W1 = M_OD_t[j-7,..., i :i+1 ]# (N,N,1) 前一周，前1个时间段，预测时间段前出闸，(N,N,3)
            # odt_W1 = np.expand_dims(odt_W1,axis=0)# (1,N,N,1)

            x_P_after_W = np.sum(M_OD[j - 7, i - P:i, ..., i:], axis=-1)  # 前7天，前P个时间段，预测时间段后出闸，(P,N,N)


            all_data[5].append(x_P_before)
            all_data[6].append(x_P_after_W)
            all_data[7].append(odt_R123)
            all_data[8].append(odt_W123)



    data_names_1 = ['samples_P_C','samples_D_C', 'samples_W_C', 'labels_C','In_C']
    data_names_2 = ['samples_P_before_C', 'samples_P_after_W_C', 'samples_P_before_odtR123_C','samples_P_before_odtW123_C']
    data_names = data_names_1 + data_names_2
    all_data = process_all(args, all_data, data_names,data_mode)

    save_feature_data(args, all_data, data_names,data_mode)
    return all_data, data_names

def sequence_OD_data_inout_flow(args, M_OD, dayofweek,data_mode,out_flow):
    # 前P个时间段,前D天,前W周
    # P = int(args.input_steps[0])
    P = 3#这里直接写入一个前面3个时刻的 In Out Flow
    D = int(args.input_steps[1])
    W = int(args.input_steps[2])
    Days, T1, N, N, T2 = M_OD.shape
    M = np.sum(M_OD, axis=-1)  # (D,T1,N,N)
    M = M.astype(np.float32)

    M1= M
    if args.data_type=="In" or args.data_type=="Out":
        M = np.expand_dims(np.sum(M_OD,axis=-1), axis=-1)  # (Days, T1, N, 1)
    Time = add_time(T1,N,dayofweek)  # (D,T1,N,T)
    M = np.concatenate([M, Time], axis=-1)  # (Days, T1, N, F+T)
    M = M.astype(np.float32)
    all_data = [[] for x in range(5+1)]
    for j in range(Days):
        # print(j)
        if j - 7 * W < 0: continue
        weeks = [j - 7 * w for w in range(1, W+1)] #[j-7,...,j-7W]
        if j - D < 0: continue
        for i in range(T1):
            if i - P < 0: continue
            y = M[j, i, ...]  # 第j天第i个时间段，(N,F)
            x_D = M[j-D:j,i,...] # 前D天第i个时间段，(D,N,F)
            x_W = M[weeks,i,...] # 前W周同一天第i个时间段(W,N,F)
            x_P = M[j,i-P:i, ...]  # 同一天，前P个时间段，全天候出闸，(P,N,F)
            In_P = np.sum(M1[j, i-P:i, ...], axis=-1)  # 同一天，前P个时间段，(P,N,)
            out_P = out_flow[j, i - P:i, ...]  # 同一天，前P个时间段，(P,N,)
            all_data[0].append(x_P)  # (M,P,N,F)
            all_data[1].append(x_D)  # (M,D,N,F)
            all_data[2].append(x_W)  # (M,W,N,F)
            all_data[3].append(y)  # (M,N,F)
            all_data[4].append(In_P) #(M,P,N,)
            all_data[5].append(out_P)  # (M,P,N,)



    data_names_1 = ['samples_P','samples_D', 'samples_W', 'labels','In','Out']
    data_names = data_names_1
    all_data = process_all(args, all_data, data_names,data_mode)

    save_feature_data(args, all_data, data_names,data_mode)
    return all_data, data_names

def sequence_OD_data_PDW(args, M_OD, dayofweek,data_mode):
    # 前P个时间段,前D天,前W周
    P = int(args.input_steps[0])
    D = int(args.input_steps[1])
    W = int(args.input_steps[2])
    Days, T1, N, N, T2 = M_OD.shape
    M = np.sum(M_OD, axis=-1)  # (D,T1,N,N)
    if args.data_type=="In" or args.data_type=="Out":
        M = np.expand_dims(np.sum(M_OD,axis=-1), axis=-1)  # (Days, T1, N, 1)
    Time = add_time(T1,N,dayofweek)  # (D,T1,N,T)
    M = np.concatenate([M, Time], axis=-1)  # (Days, T1, N, F+T)

    all_data = [[], [], [], []]
    for j in range(Days):
        if j - 7 * W < 0: continue
        weeks = [j - 7 * w for w in range(1, W+1)] #[j-7,...,j-7W]
        if j - D < 0: continue
        for i in range(T1):
            if i - P < 0: continue
            y = M[j, i, ...]  # 第j天第i个时间段，(N,F)
            x_D = M[j-D:j,i,...] # 前D天第i个时间段，(D,N,F)
            x_W = M[weeks,i,...] # 前W周同一天第i个时间段(W,N,F)
            x_P = M[j,i-P:i, ...]  # 同一天，前P个时间段，全天候出闸，(P,N,F)
            all_data[0].append(x_P)  # (M,P,N,F)
            all_data[1].append(x_D)  # (M,D,N,F)
            all_data[2].append(x_W)  # (M,W,N,F)
            all_data[3].append(y)  # (M,N,F)
    data_names = ['samples_all','samples_D', 'samples_W', 'labels']
    all_data = process_all(args, all_data, data_names,data_mode)
    save_feature_data(args, all_data, data_names,data_mode)
    return all_data, data_names

'''
功能：
输入：
原始数据=>(Days, T1, N, N) 
dayofweek => 周特征
'''
def sequence_OD_data_P(args, M_OD, dayofweek,data_mode):
    # 前P个时间段,前D天,前W周
    P = int(args.input_steps[0])
    D = int(args.input_steps[1])
    W = int(args.input_steps[2])
    Days, T1, N, N, T2 = M_OD.shape
    Time = add_time(T1, N, dayofweek)  # (D,T1,N,T)
    all_data = [[], []]
    for j in range(Days):
        if j - 7 * W < 0: continue
        if j - D < 0: continue
        for i in range(T1): # 要预测的时刻是i
            if i - P < 0: continue
            x_P_before = np.sum(M_OD[j,i-P:i,...,:i], axis=-1)  # 同一天，前P个时间段，预测时间段前出闸，(P,N,N)
            x_P_after = np.sum(M_OD[j,i-P:i,...,i:], axis=-1)  # 同一天，前P个时间段，预测时间段后出闸，(P,N,N)
            time = Time[j,i-P:i, ...]  # (P,N,3)
            x_P_before = np.concatenate([x_P_before,time],axis=-1)  # (P,N,N+3)
            x_P_after = np.concatenate([x_P_after,time],axis=-1)  # (P,N,N+3)
            all_data[0].append(x_P_before)
            all_data[1].append(x_P_after)
    data_names = ['samples_P', 'samples_after']
    all_data = process_all(args, all_data, data_names,data_mode)
    save_feature_data(args, all_data, data_names,data_mode)
    return all_data, data_names

'''
用前P个时刻预测下一个时刻
(T,N) => x = (M,P,N)  y=(M,N)
'''
def sequence_OD_data_P_VAR(args, log, M, P,data_mode):
    T, F = M.shape
    samples = []
    labels = []
    for i in range(T):
        if i + P >= T: break
        x = np.expand_dims(M[i:i + P],axis=0)  # (P,F)
        y = np.expand_dims(M[i + P],axis=0)  # (F)
        samples.append(x)  # (M,P,F)
        labels.append(y)  # (M,F)

    all_data = [samples, labels]
    data_names = ['samples', 'labels']
    all_data = process_all(args, all_data, data_names,data_mode)
    return all_data, data_names


'''
功能：求其元素级别的均值与方差
保存好均值与方差,npz格式，key是"mean"，"std"；
有OD级别和总流量级别
'''
def mean_std_save(args,log, M, data_mode):
    M_OD = np.sum(M, axis=-1)  # (D,T1,N,N)
    mean, std = np.mean(M_OD), np.std(M_OD) #(M,T,N,N)
    path = os.path.join(args.dataset_dir, args.model, data_mode, 'mean_std_OD.npz')
    np.savez_compressed(path, mean=mean, std=std)
    utils.log_string(log, 'OD=>mean:{},std:{}'.format(mean, std))
    M_IN = np.sum(M_OD,axis=-1) #(M,T,N)
    mean, std = np.mean(M_IN), np.std(M_IN)
    path = os.path.join(args.dataset_dir, args.model, data_mode, 'mean_std_Flow.npz')
    np.savez_compressed(path, mean=mean, std=std)
    utils.log_string(log, 'OD=>mean:{},std:{}'.format(mean, std))
#################################### Data process for Models : Comomon#############################################
'''
功能：装载均值和标准差
'''
def load_mean_std(args):
    # 装载均值和标准差
    path = os.path.join(args.dataset_dir, 'all_models', args.data_mode, 'mean_std_OD.npz')
    statistic = np.load(path)
    mean, std = statistic['mean'], statistic['std']
    return mean, std

'''
参数：标准化数据
'''
def standarize(args, samples, normalize=True):
    mean, std = load_mean_std(args)
    # 标准化
    if normalize:
        if args.model not in ["TR","TRs"]:
            samples = (samples - mean) / std
        else:
            samples[..., :-2] = (samples[..., :-2] - mean) / std # 最后三维是时间特征，避开
    return samples

'''
功能：打乱数据集(固定了seed)
输入一个list，list中每个元素是一个数据集，形状是多维数组，按照第一个维度将数据集打乱
'''
def shuffle(data):
    results = []
    sample_num = data[0].shape[0]
    #per = list(np.random.RandomState(seed=42).permutation(sample_num)) # 固定seed
    per = list(np.random.permutation(sample_num)) # 随机划分
    for i in range(len(data)):
        results.append(data[i][per,...])
    return results

'''
'''
def proportion(log, data, pro=0.1):
    num_sample = int(np.floor(data[0].shape[0] * pro))
    num_data = len(data)
    Data = []
    for i in range(num_data):
        Data.append(np.copy(data[i]))
    for i in range(num_data):
        Data[i] = Data[i][:num_sample,...]
        utils.log_string(log, '%s=>shape: %s' % (i, Data[i].shape))
    return Data

'''
划分训练集/验证集/测试集
'''
def data_split(args, log, data):
    batch_num = data[0].shape[0]
    train_num = round(args.train_ratio * batch_num)
    val_num = round(args.val_ratio * batch_num)
    x = len(data)
    trains = []
    vals = []
    tests = []
    for i in range(x):
        train = data[i][:train_num, ...]
        val = data[i][train_num:train_num+val_num, ...]
        test = data[i][train_num+val_num:, ...]
        trains.append(train)
        vals.append(val)
        tests.append(test)
        utils.log_string(log, '%s=>train: %s\tval: %s\ttest: %s' % (i, train.shape, val.shape, test.shape))
    return [trains, vals, tests]

'''
功能：划分batch
对data列表中每个元素的第一个维度进行batch划分，不够一个batch的丢弃掉
'''
def batch_split(args, data):
    x = len(data)
    sample_num = data[0].shape[0]
    results=[]
    for i in range(x):
        batch_num = sample_num // args.Batch_Size
        sample_num = sample_num - sample_num % args.Batch_Size
        t = data[i][:sample_num, ...]
        t = np.stack(np.split(t, batch_num, axis=0), axis=0)
        t = t.astype(np.float32)
        results.append(t)
    return results

'''
3- 保存训练集/验证集/测试集
'''
def save_dataslipt(args, data_dir, data):
    # for i in range(len(data)):
    #     data[i] = data[i].astype(np.float32)
    types = ['train', 'val', 'test']
    for i in range(len(types)):
        file_name = os.path.join(data_dir, types[i])
        if args.model in ["ANN","GCN","RNNs","TR","P","ConvLSTM"]:
            np.savez_compressed(file_name, data[i][0], data[i][1])
        elif args.model in ["TRs",'CASCNN']:
            np.savez_compressed(file_name, data[i][0], data[i][1], data[i][2], data[i][3])
        elif args.model in ["GEML"]:
            np.savez_compressed(file_name, data[i][0], data[i][1], data[i][2])
        elif args.model in ['Com']:
            np.savez_compressed(file_name, data[i][0], data[i][1], data[i][2],data[i][3], data[i][4],data[i][5])
        # elif args.model in ['Com']:
        #     np.savez_compressed(file_name, data[i][0], data[i][1], data[i][2], data[i][3], data[i][4])
        else:

            print("No such model! => ",args.model)
            raise ValueError

'''
根据模型去掉时间特征
'''
def filter_time(args, data, save_time=False):
    if args.model not in ["TR","TRs"]:
        data[1] = data[1][...,:-3] # 去掉输入的时间特征
    elif args.model in ["TR"]:
        data[1] = data[1][...,:-1] # 去掉输入的最后一维时间特征
    elif args.model in ["TRs"]:
        data[1] = data[1][...,:-1] # 去掉输入的最后一维时间特征
        data[2] = data[2][...,:-1] # 去掉输入的最后一维时间特征
        data[3] = data[3][...,:-1] # 去掉输入的最后一维时间特征
    else:
        print("Wrong!")
    if save_time:
        time = data[0][..., -3:]
        data.append(time)
    data[0] = data[0][..., :-3]  # 去掉输出的时间特征
    return data
'''
3- 数据处理全流程
'''
def data_process(args):
    print(args)
    # 按输入模式，产生文件夹
    log_dir, save_dir, data_dir,  dir_name = utils.create_dir_PDW(args)
    log = utils.create_log(args, args.data_log)
    # 装载数据：[(M,N,N),(M,T1,N,N),(M,T2,N,N),...]
    data = load(args, log, dir_name)
    # 根据模型去掉时间特征
    data = filter_time(args, data)
    # 打乱数据集
    data = shuffle(data)
    # 取部分数据
    data = proportion(log, data, args.proportion)
    if args.model in ["GEML"]:
        # 产生距离图
        path = os.path.join(args.dataset_dir, 'original', 'graph_geo.npz')
        # print(path)
        A = np.load(path)['arr_0'].astype(np.float32)  # (N,N)
        graph_geo = graph_common.distance_graph_GEML(A)  # (N,N)
        path = os.path.join(data_dir, 'graph_geo.npz')
        # print(data_dir)
        np.savez_compressed(path, graph=graph_geo)
        # 产生上下文图
        graph_sem = graph_common.semantic_graph_GEML(data[1])  # (M,P,N,N)
        data.append(graph_sem)

        # exit()
    if args.model == "GCN":
        graph_common.GCN_graph(args, data_dir)  # (N,N)
    if args.model in ["ANN","GCN"] :
        data= ANN_data_vector(log, data) # 预测向量
        #data= ANN_data_scalar(log, data)  # 预测标量
    if args.model in ["RNNs"]:
        data = RNNs_data_vector(log, data)  # 预测向量
        #data = RNNs_data_matrix(log, data)  # 预测矩阵

    # 标准化数据:只标准化输入X
    data[1] = standarize(args, data[1], normalize=args.Normalize)
    if args.model == "TRs":
        data[2] = standarize(args, data[2], normalize=args.Normalize)
        data[3] = standarize(args, data[3], normalize=args.Normalize)
    if args.model == "CASCNN":
        data[1] = data[1].transpose(0, 2, 3, 1)  # 1 (M, N , N, P)
        data[2] = np.expand_dims(data[2].transpose(0, 2, 1), axis=2)  # 1 (M, N , N, P)
        data[2] = (data[2] - np.mean(data[2])) / np.std(data[2])  # 2 (M,N,1,P)
        data[3] = np.expand_dims(data[3].transpose(0, 2, 1), axis=2)
        data[3] = (data[3] - np.mean(data[3])) / np.std(data[3])  # 3 (M,N,1,P)
    if args.model == "Com":
        if args.Normalize:
            print("Com normalize")
            for num in range(2,len(data)):
                data[num] = (data[num]-np.mean(data[1]))/np.std(data[1])#


    # 检查多维数组是否有nan,inf
    for d in data:
        utils.check_inf_nan(d)

    # 划分batch
    data = batch_split(args, data) # M=>(V,B)
    # 划分训练集/验证集/测试集
    data = data_split(args, log, data)
    # 保存数据
    save_dataslipt(args, data_dir, data)
    utils.log_string(log, 'data_type:%s, prediction type:%s'%(args.data_type, args.output_type))
    utils.log_string(log, 'Finish\n')
    return data

def test_data_process(args, log, dir_name, data_dir):
    # 装载数据：[(M,N,N),(M,T1,N,N),(M,T2,N,N),...]
    data = load(args, log, dir_name)
    # 根据模型去掉时间特征:x,y,time
    Time =  data[0][...,-3:]
    data = filter_time(args, data)
    samples = data[1]
    if args.model in ["GEML"]:
        # 产生上下文图
        graph_sem = graph_common.semantic_graph_GEML(samples)  # (M,P,N,N)
        data.append(graph_sem)
    if args.model == "GCN":
        graph_common.GCN_graph(args, data_dir)  # (N,N)
    if args.model in ["ANN", "GCN"]:
        data = ANN_data_vector(log, data)  # 预测向量
    if args.model in ["RNNs"]:
        data = RNNs_data_vector(log, data)  # 预测向量

    # 标准化数据:只标准化输入X
    data[1] = standarize(args, data[1], normalize=args.Normalize)
    if args.model == "TRs":
        data[2] = standarize(args, data[2], normalize=args.Normalize)
        data[3] = standarize(args, data[3], normalize=args.Normalize)
    if args.model == "CASCNN":
        data[1] = data[1].transpose(0, 2, 3, 1)  # 1 (M, N , N, P)
        print(data[2].shape)
        data[2] = np.expand_dims(data[2].transpose(0, 2, 1), axis=2)  # 1 (M, N , N, P)
        data[2] = (data[2] - np.mean(data[2])) / np.std(data[2])  # 2 (M,N,1,P)
        data[3] = np.expand_dims(data[3].transpose(0, 2, 1), axis=2)
        data[3] = (data[3] - np.mean(data[3])) / np.std(data[3])  # 3 (M,N,1,P)

    # 检查多维数组是否有nan,inf
    for d in data:
        utils.check_inf_nan(d)
    utils.log_string(log, 'Data Loaded Finish...\n')
    return data,Time

'''
3- 主模型装载数据
装载训练集/测试集/验证集，标准差/均值
'''
def load_data(args, log, data_dir, yes=True):
    utils.log_string(log, 'loading data...')
    types = ['train','val','test']
    results = []
    for type in types:
        path = os.path.join(data_dir, '%s.npz'%(type))
        data = np.load(path)
        dict = {}
        for j in range(len(data)):
            name = "arr_%s" % j
            dict[name] = data[name]
            utils.log_string(log, '%s=>%s shape: %s,type:%s' % (type, j, dict[name].shape,dict[name].dtype))
        results.append(dict)
    # 装载标准差和均值
    if yes:
        mean, std = load_mean_std(args)
    else:
        mean, std = 0.0, 1.0 # 数据不标准化
    utils.log_string(log,'mean:{},std:{}'.format(mean, std))
    return results, mean, std

# '''
# 3- npz转换为dict
# '''
# def npz_to_dict(log, data):
#     results = []
#     for i in range(len(data)):
#         dict = {}
#         for j in range(len(data[i])):
#             name = "arr_%s"%j
#             dict[name] = data[i][name]
#             utils.log_string(log, '%s=>%s shape: %s' % (i,j, dict[name].shape))
#         results.append(dict)
#     return results

#################################### Data process for Models : Single#############################################
def Regression_random_data():
    sample_num = 500
    x_fea, y_fea = 20, 10
    X = np.random.normal(size=(sample_num, x_fea))
    Y = np.random.normal(size=(sample_num, x_fea))
    return [X,Y]

def generate_data_random_GEML():
    B,T, N, F_in, F_out = 5,13,10,10,10
    graph_geo = np.random.normal(size=(N, N)).astype(np.float32)
    M = [5, 2, 3]
    data = []
    for i in range(3):
        graph_sem = np.random.normal(size=(M[i],B,T, N, N)).astype(np.float32)
        samples = np.random.normal(size=(M[i],B,T, N, F_in)).astype(np.float32)
        M_Labels = np.random.normal(size=(M[i],B,N, F_out)).astype(np.float32)
        P_Labels = np.random.normal(size=(M[i],B,N, 1)).astype(np.float32)
        Q_Labels = np.random.normal(size=(M[i],B,N, 1)).astype(np.float32)
        dict = {'x':samples,'y':M_Labels,'yp':P_Labels,'yq':Q_Labels,'graph':graph_sem}
        data.append(dict)
    return graph_geo, data

#数据变形:X=(M,T,N,N)=>(M,N,T,N)=>(MN,TN), Y=(M,N,N)=>(MN,N)
def Regression_data_vector(data):
    X = data[1] #(M,T,N,N)
    Y = data[0] #(M,N,N)
    M, T, N, N = X.shape
    X = X.swapaxes(1, 2) #(M,T,N,N)=>(M,N,T,N)
    X = np.reshape(X, [M*N,T*N]) #(MN,TN)
    Y = np.reshape(Y, [M*N,N]) #(MN,N)
    data = [X,Y]
    return data

# scalar 的预测效果是最差的
#数据变形:X=(M,T,N,N)=>(M,N,N,T)=>(M*N*N,T),Y=(M,N,N)=>(M*N*N,1)
def Regression_data_scalar(data):
    X = data[1] #(M,T,N,N)
    Y = data[0] #(M,N,N)
    M, T, N, N = X.shape
    X = np.squeeze(np.stack(np.split(X, T, axis=1),axis=-1),axis=1)
    X = np.reshape(X, [M*N*N,T]) #(MNN,T)
    Y = np.reshape(Y, [M*N*N,1]) ##(MNN,1)
    data = [Y,X]
    return data

# 预测的最小单元是单个站点在某个站点的出闸人数: X=(...,T,N,N)=>(...,N,N,T),Y=(...,N,N,1)
def ANN_data_scalar(log, data):
    data[1] = np.expand_dims(data[1], axis=-1) # (...,T,N,N,1)
    x = np.split(data[1],data[1].shape[-4],axis=-4) # (...,1,N,N,1)
    data[1] = np.squeeze(np.concatenate(x,axis=-1)) # (...,1,N,N,T)=>(...,N,N,T)
    data[0] = np.expand_dims(data[0], axis=-1) #(...,N,N)=>(...,N,N,1)
    utils.log_string(log, "x shape:%s y shape:%s"%(str(data[1].shape), str(data[0].shape)))
    return data

# 预测的最小单元是单个站点的出闸分布：X=(...,T,N,N)=>(...,N,NT),Y=(...,N,N)
def ANN_data_vector(log, data):
    data[1] = np.squeeze(np.concatenate(np.split(data[1], data[1].shape[-3], axis=-3), axis=-1))
    utils.log_string(log, "x shape:%s y shape:%s"%(str(data[1].shape), str(data[0].shape)))
    return data

# X:(...,T,N,N)=>(...,N,T,N)=>(VN,...,T,N), Y:(V,...,N,N)=>(V*N,...,N)
def RNNs_data_vector(log, data):
    # print("RNNs_data_vector")
    # print(data[0].shape)
    # print(data[1].shape)
    # exit()
    # data[1] = data[1].swapaxes(-3,-2) #(...,N,T,N)
    # x = np.split(data[1], data[1].shape[-3], axis=-3) #(...,1,T,N)
    # data[1] = np.squeeze(np.concatenate(x, axis=0)) # (MN,...,T,N)
    # y = np.split(data[0], data[0].shape[-2], axis=-2) #(M,...,1,N)
    # z = np.concatenate(y, axis=0)
    # data[0] = np.squeeze(z) # (MN,...,N)
    # utils.log_string(log, "x shape:%s y shape:%s"%(str(data[1].shape), str(data[0].shape)))
    # return data

    M, T, N, _ = data[1].shape
    data[1] = data[1].swapaxes(1,2) #(M,N,T,N)
    data[1] = np.reshape(data[1], (M*N, T, N))
    data[0] = np.reshape(data[0][...,:N], (M*N, N))
    utils.log_string(log, "x shape:%s y shape:%s"%(str(data[0].shape), str(data[1].shape)))
    return data

def RNNs_data_vector_inv(data):

    N = data.shape[-1]
    M = int(data.shape[0]/N)
    tmp = np.reshape(data, (M,N, N))
    return tmp

def RNNs_data_matrix(log, data):
    x = np.split(data[1], data[1].shape[-1], axis=-1) #(...,T,N,1)
    data[1] = np.squeeze(np.concatenate(x, axis=-2)) # (M,...,T,N*N)
    y = np.split(data[0], data[0].shape[-1], axis=-1) #(M,...,N,1)
    data[0] = np.squeeze(np.concatenate(y, axis=-2)) # (M,...,N*N)
    utils.log_string(log, "x shape:%s y shape:%s"%(str(data[1].shape), str(data[0].shape)))
    return data
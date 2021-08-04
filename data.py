from libs import utils, data_common,para

def main(args):
    dir_name, data_path, log = utils.path(args) # 创建存放数据和日志的文件夹
    M = data_common.load_original_data(args, log) # 原始数据格式是NPZ,key是"matrix"
    data_common.mean_std_save(args,log, M) # 存储标准差和均值
    dayofweek = data_common.week_transform(args, log) #时间特征，(Days, T1, N, T)
    data_mode = "data1"
    data_common.sequence_OD_data_PDW(args, M, dayofweek,data_mode) # 产生P,D,W数据
    if args.data_type=='InOD':
        data_common.sequence_OD_data_P(args, M, dayofweek,data_mode) # 产生before, after数据

    data_mode2 = "data2"
    weekend = []
    weekdays = []
    for i in range(dayofweek.shape[0]):
        if dayofweek[i] < 5:
            weekdays.append(i)
        else:
            weekend.append(i)
    _, T1, N, _, _ = M.shape
    Time = data_common.add_time(T1, N, dayofweek)  # (D,T1,N,T)
    all_data = data_common.sequence_OD_data_PDW2(args, M[weekdays], Time[weekdays], 5)  # 产生P,D,W数据
    all_data2 = data_common.sequence_OD_data_PDW2(args, M[weekend], Time[weekend], 2)  # 产生P,D,W数据
    data_names = ['samples_all', 'samples_D', 'samples_W', 'labels', 'samples_P', 'samples_after']
    for i in range(len(all_data2)):
        all_data[i] = all_data[i] + all_data2[i]
    all_data = data_common.process_all(args, all_data, data_names)
    data_common.save_feature_data(args, all_data, data_names, data_mode2)
    utils.log_string(log, 'Finish：%s\n' % args.data_type)
    utils.log_string(log, 'Finish：%s\n' % args.data_type)

if __name__ == "__main__":
        parser = para.original_data_para()
        args = parser.parse_args()
        main(args)
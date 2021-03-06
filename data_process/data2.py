import libs.utils, libs.data_common,libs.para

def main(args):

    print(args)
    data_mode = args.data_mode
    dir_name, data_path, log = libs.utils.path(args,data_mode) # 创建存放数据和日志的文件夹
    M = libs.data_common.load_original_data(args, log) # 原始数据格式是NPZ,key是"matrix"
    libs.data_common.mean_std_save(args,log, M, data_mode) # 存储标准差和均值
    dayofweek = libs.data_common.week_transform(args, log) #时间特征，(Days, T1, N, T)
    weekend = []
    weekdays = []
    for i in range(dayofweek.shape[0]):
        if dayofweek[i]<5: weekdays.append(i)
        else:weekend.append(i)
    Days, T1, N, _, _ = M.shape
    Time = libs.data_common.add_time(T1, N, dayofweek)  # (D,T1,N,T)
    all_data = libs.data_common.sequence_OD_data_PDW2(args, M[weekdays], Time[weekdays], 5)  # 产生P,D,W数据
    all_data2 = libs.data_common.sequence_OD_data_PDW2(args, M[weekend], Time[weekend], 2)  # 产生P,D,W数据
    data_names = ['samples_all', 'samples_D', 'samples_W', 'labels', 'samples_P', 'samples_after']
    for i in range(len(all_data2)):
        all_data[i] = all_data[i]+all_data2[i]

    all_data = libs.data_common.process_all(args, all_data, data_names,data_mode)
    libs.data_common.save_feature_data(args, all_data, data_names,data_mode)
    libs.utils.log_string(log, 'Finish：%s\n' % args.data_type)

if __name__ == "__main__":
        parser = libs.para.original_data_libs.para()
        args = parser.parse_args()
        main(args)
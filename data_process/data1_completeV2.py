import libs.utils, libs.data_common,libs.para

def main(args):
    print(args)
    data_mode = args.data_mode
    dir_name, data_path, log = libs.utils.path(args,data_mode) # 创建存放数据和日志的文件夹
    M = libs.data_common.load_original_data(args, log) # 原始数据格式是NPZ,key是"matrix"
    libs.data_common.mean_std_save(args,log, M,data_mode) # 存储标准差和均值
    dayofweek = libs.data_common.week_transform(args, log) #时间特征，(Days, T1, N, T)

    libs.data_common.sequence_OD_data_all_C_V2(args, M, dayofweek,data_mode)  # 产生P,D,W数据

    # libs.data_common.sequence_OD_data_PDW(args, M, dayofweek,data_mode) # 产生P,D,W数据
    # if args.data_type=='InOD':
    #     libs.data_common.sequence_OD_data_P(args, M, dayofweek,data_mode) # 产生before, after数据
    # libs.utils.log_string(log, 'Finish：%s\n' % args.data_type)

if __name__ == "__main__":
        parser = libs.para.data_complete_V2_data_libs.para()
        args = parser.parse_args()
        main(args)
from libs import utils, data_common,para
import os
import numpy as np
def main(args):
    print(args)
    data_mode = args.data_mode
    dir_name, data_path, log = utils.path(args,data_mode) # 创建存放数据和日志的文件夹

    data_file = os.path.join(args.dataset_dir, 'original', args.data_out_file)
    M_out = np.load(data_file)["matrix"]
    out_flow = np.sum(np.sum(M_out,axis=-1),axis=-1)# (D,T,N)
    out_flow = out_flow.astype(np.float32)
    data_file = os.path.join(args.dataset_dir, 'original', args.data_in_file)
    M_in = np.load(data_file)["matrix"].astype(np.float32)

    data_common.mean_std_save(args,log, M_in,data_mode) # 存储标准差和均值
    dayofweek = data_common.week_transform(args, log) #时间特征，(Days, T1, N, T)

    data_common.sequence_OD_data_inout_flow(args, M_in, dayofweek,data_mode,out_flow)  # 产生P,D,W数据

if __name__ == "__main__":
        parser = para.original_data_CASCNN_para()
        args = parser.parse_args()
        main(args)
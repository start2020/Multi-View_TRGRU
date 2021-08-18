import numpy as np
import os
import pandas as pd
def get_str_index(str,char):
    return [i for i, x in enumerate(str) if x == char]

# /root/software/yes/envs/tf/bin/python cat_res.py

mode_list = ["P3D0W0","P0D3W0","P0D0W3"]
# mode_list = ["P3D2W1","P3D3W1","P3D1W1"]
mode_list = ["P5D4W2"]
# mode_list = ["P5D4W2"]
# mode_list = ["P3D2W1-sche3"]
# max_val = 8
#
# mode_list = []
# for i in range(1,max_val):
#     mode_list .append("P{}D0W0".format(i))
# for i in range(1,max_val):
#     mode_list .append("P0D{}W0".format(i))
# for i in range(1,8):
#     mode_list .append("P0D0W{}".format(i))
# for p in range(3,max_val):
#     for w in range(3,max_val):
#         for d in range(1,4):
#             mode_list.append("P{}D{}W{}".format(p,w,d))

# model_name_list = ["HA","Lasso","Ridge","ANN","RNNs","ConvLSTM","GCN","GEML","TR","CASCNN"]
model_name_list = ["TRs"]
# citys = ["hz-CAS","sh-CAS","sz-CAS"]
# citys = ["hz-Mat","sh-Mat","sz-Mat"]
citys = ["sz-2","sh"]
# citys = ["sz-2","sh","hz"]
metrics = ['MAE', 'RMSE', 'WMAPE', 'SMAPE']
df_empty = pd.DataFrame(columns=['mode', 'model']+metrics)
for city in citys:
    for mode in mode_list:
        for model_name in model_name_list:

            print(city, mode,model_name)
            res = []
            # if mode=="P0D3W0":
            #     name = 'D'
            # if mode=="P3D0W0":
            #     name = 'P'
            # if mode=="P0D0W3":
            #     name = 'W'
            # abs_file_name = "{}/{}/log/{}_result_log".format(city, mode, model_name)
            # abs_file_name = "{}/{}/log/{}_result_log_runit512_N".format(city, mode, model_name)
            # abs_file_name = "{}/{}/log/{}_result_log_1.0".format(city,mode,model_name)
            # abs_file_name = "{}/{}/log/{}_result_log_sche3".format(city, mode, model_name)
            # abs_file_name = "{}/{}/log/{}_result_log".format(city, mode, model_name)
            # abs_file_name = "{}/{}-ICP/log/{}_result_log".format(city, mode, model_name)
            # abs_file_name = "{}/{}/log/{}_result_log_CP".format(city, mode, model_name)

            abs_file_name = "{}/{}/log/{}_result_log_no_TE".format(city, mode, model_name)
            # abs_file_name = "{}/{}/log/{}_result_log_no_SE".format(city, mode, model_name)
            # abs_file_name = "{}/{}/log/{}_result_log_no_SETE".format(city, mode, model_name)
            # abs_file_name = "{}/{}/log/{}_result_log_no_RNN".format(city, mode, model_name)
            if os.path.isfile(abs_file_name):

                with open(abs_file_name, 'r',encoding='utf-8') as fin:
                    for line in fin.readlines():
                        line = line[:-1]  # 去掉\n
                        line_split_list = line.split('\t')
                        print(line_split_list)
                        tmp = []
                        if line_split_list[0] == "MAE":
                            for i in range(int(len(line_split_list)/2)):
                                tmp.append(float(line_split_list[2*i+1]))
                        else:
                            continue
                        res.append(tmp)

                if len(res)>0:
                    res = np.array(res)
                    res_one_list = []
                    for i in range(res.shape[1]):
                        message = "%.4f±%.4f"%(np.mean(res[:,i]),np.std(res[:,i]))
                        print(metrics[i],message)
                        res_one_list.append(message)
                    # df_empty.loc[mode+model_name] = [mode,model_name,'1.6934', '6.1350',  '0.5158', '0.3226']
                    # char_num = ""
                    # for i in range(3,max_val):
                    #     if str(i) in mode:
                    #         char_num = str(i)
                    #
                    # if not char_num.isdigit():
                    #     print(char_num)
                    #     raise TypeError
                    # str_list = get_str_index(mode, char_num)
                    # mode_str=""
                    # for i in str_list:
                    #     mode_str +=mode[i-1:i+1]
                    # print(res_one_list)
                    # df_empty.loc[city + mode_str + model_name[:3]] = [mode, model_name]+res_one_list

                    df_empty.loc[city +"_"+ mode + "_" + model_name[:3]] = [mode, model_name] + res_one_list
                else:
                    print("len too short")
            else:
                print("file not found",abs_file_name)
df_empty.to_excel("./res.xlsx")

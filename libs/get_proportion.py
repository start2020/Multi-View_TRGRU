import numpy as np
import xlrd
import pandas as pd

# P表示P步预测1步
# t是range(P)的其中一个值
# time_slot:时间粒度
# Tod_ave : 入闸站点到出闸站点的平均时间
# proportion: 入闸站点到出闸站点在预测时间P之前出去的比例(概率)
def get_a_od_pro(P,t,time_slot,Tod_ave):
    if Tod_ave<(P-t-1)*time_slot:
        proportion = 1.0
    elif Tod_ave>(P-t)*time_slot:
        proportion = 0.0
    else:
        proportion = ((P-t)*time_slot -Tod_ave)/time_slot
    print(P,t,time_slot,Tod_ave,proportion)
    return proportion



# P表示P步预测1步
# t是range(P)的其中一个值
# time_slot:时间粒度
# Tod_ave : 入闸站点到出闸站点的平均时间,维度(N*N)
# proportion: 入闸站点到出闸站点在预测时间P之前出去的比例(概率)
def get_ods_pro(P,t,time_slot,Tod_ave):

    N = Tod_ave.shape[0]
    # proportion = np.zeros(shape=[N,N])
    proportion = []
    for i in range(N):
        for j in range(N):
            proportion.append(get_a_od_pro(P, t, time_slot, Tod_ave[i][j]))
            # proportion[i][j] = get_a_od_pro(P, t, time_slot, Tod_ave[i][j])
    return np.reshape(proportion,(N,N))

# N = 4
# P= 3
# t = 1
# time_slot = 30
# Tod_ave = np.around(np.random.rand(N,N) * 100)
# for i in range(N):
#     Tod_ave[i][i] = 10
#     for j in range(N):
#         if Tod_ave[i][j] < 10:
#             Tod_ave[i][j] = 10
#
# Tod_ave = np.triu(Tod_ave)
# Tod_ave += Tod_ave.T - np.diag(Tod_ave.diagonal())
# print(Tod_ave)
#
# proportion = get_ods_pro(P,t,time_slot,Tod_ave)
# print(proportion)



excel_file = xlrd.open_workbook('sz/data/shortest.xlsx')
sheet = excel_file.sheet_by_index(0)#索引的方式，从0开始
nrows = sheet.nrows#行
ncols = sheet.ncols#列
print(type(sheet))
print(nrows)
print(ncols)

Tod_ave = np.zeros(shape=[nrows-1,ncols-1])
N = Tod_ave.shape[0]
for i in range(N):
    for j in range(N):
         Tod_ave[i][j] = sheet.cell(i+1, j+1).value
print(Tod_ave)

# df = pd.read_excel('sz/data/shortest.xlsx')  # 直接默认读取到Excel的第一个表单
# print(df)
# min_time = df.values()
# print(min_time)


P= 3
t = 1
time_slot = 30
proportion = get_ods_pro(P,t,time_slot,Tod_ave)
print(proportion)
proportion_df =pd.DataFrame(proportion)
proportion_df.to_excel("sz/data/proportion.xlsx")


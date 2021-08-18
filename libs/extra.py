import numpy as np
import os
import datetime
from libs import utils
import pickle

import os

os.system('rm -rf {}'.format(save_dir + '/*'))  # 删除保存的模型,用脚本的时候再用

'''
保存训练集/验证集/测试集
pickle方式保存的话，会占用较大的存储空间
'''
def save_dataslipt_pkl(data_dir,data, names):
    types = ['train', 'val', 'test']
    for i in range(len(types)):
        dict = {}
        file_name = os.path.join(data_dir, '%s.pkl'%types[i])
        file = open(file_name, 'wb')
        for j in range(len(names)):
            dict[names[j]]=data[i][j]
        pickle.dump(dict, file)
        file.close()
import numpy as np
import os
import datetime
from libs import utils
import pickle

import os

os.system('rm -rf {}'.format(save_dir + '/*'))  # ɾ�������ģ��,�ýű���ʱ������

'''
����ѵ����/��֤��/���Լ�
pickle��ʽ����Ļ�����ռ�ýϴ�Ĵ洢�ռ�
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
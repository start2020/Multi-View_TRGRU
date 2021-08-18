import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import libs.para, libs.main_common
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = libs.para.GCN_main(parser)
    parser = libs.para.common_para(parser)
    parser = libs.para.GCN(parser)
    args = parser.parse_args()
    libs.main_common.main(args)
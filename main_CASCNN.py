import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from libs import para, main_common
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = para.CASCNN_main(parser)
    parser = para.common_para(parser)
    parser = para.CASCNN(parser)
    args = parser.parse_args()
    main_common.main(args)
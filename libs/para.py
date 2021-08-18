# coding: utf-8
import argparse
import numpy as np


def data_complete_V2_data_para():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='hz-cp-test/')
    parser.add_argument('--data_file', default="matrix_in_30.npz")
    parser.add_argument('--day_index', default="day-index-201901.txt")
    parser.add_argument('--model', default='all_models/', type=str)
    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3, 0, 0], type=list, help="3-6")
    parser.add_argument('--data_mode', type=str, default="data1", help='data creat mode')
    return parser

def original_data_para():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='hz/')
    parser.add_argument('--data_file', default="matrix_in_30.npz")
    parser.add_argument('--day_index', default="day-index-201901.txt")
    parser.add_argument('--model', default='all_models/', type=str)
    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3, 0, 0], type=list, help="3-6")
    parser.add_argument('--data_mode', type=str, default="data1", help='data creat mode')
    return parser

def original_data_CASCNN_para():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='hz-CAS/')
    parser.add_argument('--data_in_file', default="matrix_in_30.npz")
    parser.add_argument('--data_out_file', default="matrix_out_30.npz")
    parser.add_argument('--day_index', default="day-index-201901.txt")
    parser.add_argument('--model', default='all_models/', type=str)
    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[0, 3, 0], type=list, help="3-6")
    parser.add_argument('--data_mode', type=str, default="data3", help='data creat mode')
    return parser

def original_data_Mat_para():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='hz-Mat/')
    parser.add_argument('--data_file', default="matrix_in_30.npz")
    parser.add_argument('--day_index', default="day-index-201901.txt")
    parser.add_argument('--model', default='all_models/', type=str)
    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3, 0, 0], type=list, help="3-6")
    parser.add_argument('--data_mode', type=str, default="data1", help='data creat mode')
    parser.add_argument('--previous', default=100, type=int)
    return parser

# 所有数据处理data_model.py都会有的
def common_para(parser):
    parser.add_argument('--dataset_dir', default='hz/', type=str)
    parser.add_argument('--N', default=80, type=int)
    parser.add_argument('--P', default=3, type=int)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--data_log', default='data_log', type=str)
    parser.add_argument('--Null_Val', type=float, default=np.nan, help='value for missing data')

    parser.add_argument('--Batch_Size', type=int, default=32)
    parser.add_argument('--max_grad_norm', type=float, default=5.)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--max_epoch', type=int, default=1000, help='epoch to run')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stop')
    parser.add_argument('--lr_decay_ratio', type=float, default=0.9)
    parser.add_argument('--steps', type=int, default=5, help="five learning rate")
    parser.add_argument('--min_learning_rate', type=float, default=2.0e-06)
    parser.add_argument('--base_lr', type=float, default=0.01,help='initial learning rate')
    parser.add_argument('--data_mode', type=str, default="data1", help='data creat mode')
    parser.add_argument('--continue_train', type=int, default=0, help='initial withou old model')
    parser.add_argument('--train_shuffle', type=int, default=0, help='train shuffle')
    parser.add_argument('--description', type=str, default="no description", help='parameter description or experience description')

    return parser


# HA的main
def HA_main(parser):
    parser.add_argument('--model', default='HA', type=str)
    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3,0,0], type=list, help="3-6")
    parser.add_argument('--data_names', default="labels-samples_P", type=str)
    parser.add_argument('--output_type', default='sclar', type=str, help="sclar, vector, matrix")
    parser.add_argument('--proportion', default=1.0, type=float)

    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--test_log', default='test_log', type=str)
    parser.add_argument('--test', default=0, type=int,help="Do test")
    parser.add_argument('--Times', type=int, default=5, help="the number of experiments")
    return parser

# HA的main
def Mat_main(parser):
    parser.add_argument('--model', default='Mat', type=str)
    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3,0,0], type=list, help="3-6")
    parser.add_argument('--data_names', default="labels-samples_P", type=str)
    parser.add_argument('--output_type', default='sclar', type=str, help="sclar, vector, matrix")
    parser.add_argument('--proportion', default=1.0, type=float)

    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--test_log', default='test_log', type=str)
    parser.add_argument('--test', default=0, type=int,help="Do test")
    parser.add_argument('--Times', type=int, default=1, help="the number of experiments")
    return parser

# Lasso的main
def Lasso_main(parser):
    parser.add_argument('--model', default='Lasso', type=str)

    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3,0,0], type=list, help="3-6")
    parser.add_argument('--data_names', default="labels-samples_P", type=str)
    parser.add_argument('--output_type', default='sclar', type=str, help="sclar, vector, matrix")
    parser.add_argument('--proportion', default=0.01, type=float)

    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--test_log', default='test_log', type=str)
    parser.add_argument('--test', default=1, type=int,help="Do test")
    parser.add_argument('--Times', type=int, default=5, help="the number of experiments")
    return parser

# Ridge的main
def Ridge_main(parser):
    parser.add_argument('--model', default='Ridge', type=str)

    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3,0,0], type=list, help="3-6")
    parser.add_argument('--data_names', default="labels-samples_P", type=str)
    parser.add_argument('--output_type', default='sclar', type=str, help="sclar, vector, matrix")
    parser.add_argument('--proportion', default=0.01, type=float)

    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--test_log', default='test_log', type=str)
    parser.add_argument('--test', default=0, type=int,help="Do test")
    parser.add_argument('--Times', default=5, type=int, help="the number of experiment")
    return parser



# RNNs的data和main都会有的
def Com_main(parser):
    parser.add_argument('--model', default='Com', type=str)
    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3, 0, 0], type=list, help="3-6")
    # parser.add_argument('--data_names', default="samples_P_C-samples_P_before_C-samples_P_after_W_C-samples_P_before_odt123_C-In_C-samples_P_after_C-samples_D3_C-samples_P_before_odt123_C-samples_P_before_odt0_C-samples_P_before_odt0123_C", type=str)
    parser.add_argument('--data_names',default="samples_P_C-samples_P_before_C-samples_P_after_W_C-samples_P_before_odt123_C-In_C",type=str)
    # parser.add_argument('--data_names',
    #                     default="samples_P_C-samples_P_before_C-samples_P_after_W_C-samples_P_before_odtR1_C-In_C-samples_P_before_odtD1_C",
    #                     type=str)
    # parser.add_argument('--data_names',default="samples_P_C-samples_P_before_C-In_C-samples_D3_C",type=str)
    parser.add_argument('--save_dir', default="1", type=str)
    # labels, samples_P_before_C, samples_P_after_W_C, samples_P_before_odt_C, IN
    # 'samples_all_C','samples_D_C', 'samples_W_C', 'labels_C','In_C'
    # 'samples_P_before_C', 'samples_P_after_W_C','samples_P_before_odt_C'
    parser.add_argument('--output_type', default='matrix', type=str, help="sclar, vector, matrix")
    parser.add_argument('--Normalize', type=int, default=0)
    parser.add_argument('--proportion', default=1.0, type=float)
    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--val_loss', default=1, type=int, help="1-val_loss, 0-test_loss")
    parser.add_argument('--loss_type', default=1, type=int, help="1-single, 2-multi_man, 3-multi_auto")
    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--test_log', default='test_log', type=str)
    parser.add_argument('--data', default=0, type=int,help="produce data")
    parser.add_argument('--experiment', default=1, type=int,help="Do experiments")
    parser.add_argument('--test', default=0, type=int,help="Do test")
    parser.add_argument('--Times', default=1, type=int, help="the number of experiment")
    return parser

# ANN的data和main都会有的
def ANN_main(parser):
    parser.add_argument('--model', default='ANN', type=str)

    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3, 0, 0], type=list, help="3-6")
    parser.add_argument('--data_names', default="labels-samples_P", type=str)
    parser.add_argument('--output_type', default='vector', type=str, help="sclar, vector, matrix")
    parser.add_argument('--Normalize', type=bool, default=True)
    parser.add_argument('--proportion', default=1.0, type=float)

    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--val_loss', default=1, type=int, help="1-val_loss, 0-test_loss")
    parser.add_argument('--loss_type', default=1, type=int, help="1-single, 2-multi_man, 3-multi_auto")

    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--test_log', default='test_log', type=str)
    parser.add_argument('--data', default=0, type=int,help="produce data")
    parser.add_argument('--experiment', default=0, type=int,help="Do experiments")
    parser.add_argument('--test', default=0, type=int,help="Do test")
    parser.add_argument('--Times', default=5, type=int, help="the number of experiment")
    return parser

# RNNs的data和main都会有的
def RNNs_main(parser):
    parser.add_argument('--model', default='RNNs', type=str)

    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3, 0, 0], type=list, help="3-6")
    parser.add_argument('--data_names', default="labels-samples_P", type=str)
    parser.add_argument('--output_type', default='vector', type=str, help="sclar, vector, matrix")
    parser.add_argument('--Normalize', type=bool, default=True)
    parser.add_argument('--proportion', default=1.0, type=float)

    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--val_loss', default=1, type=int, help="1-val_loss, 0-test_loss")
    parser.add_argument('--loss_type', default=1, type=int, help="1-single, 2-multi_man, 3-multi_auto")

    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--test_log', default='test_log', type=str)
    parser.add_argument('--data', default=0, type=int,help="produce data")
    parser.add_argument('--experiment', default=0, type=int,help="Do experiments")
    parser.add_argument('--test', default=0, type=int,help="Do test")
    parser.add_argument('--Times', default=5, type=int, help="the number of experiment")
    return parser

# RNNs的data和main都会有的
def ConvLSTM_main(parser):
    parser.add_argument('--model', default='ConvLSTM', type=str)

    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3, 0, 0], type=list, help="3-6")
    parser.add_argument('--data_names', default="labels-samples_P", type=str)
    parser.add_argument('--output_type', default='vector', type=str, help="sclar, vector, matrix")
    parser.add_argument('--Normalize', type=bool, default=True)
    parser.add_argument('--proportion', default=1.0, type=float)

    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--val_loss', default=1, type=int, help="1-val_loss, 0-test_loss")
    parser.add_argument('--loss_type', default=1, type=int, help="1-single, 2-multi_man, 3-multi_auto")

    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--test_log', default='test_log', type=str)
    parser.add_argument('--data', default=0, type=int,help="produce data")
    parser.add_argument('--experiment', default=1, type=int,help="Do experiments")
    parser.add_argument('--test', default=0, type=int,help="Do test")
    parser.add_argument('--Times', default=5, type=int, help="the number of experiment")
    return parser

# GCN的data和main都会有的
def GCN_main(parser):
    parser.add_argument('--model', default='GCN', type=str)

    parser.add_argument('--graph_name', type=str, default="graph_connection.npz")
    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3, 0, 0], type=list, help="3-6")
    parser.add_argument('--data_names', default="labels-samples_P", type=str)
    parser.add_argument('--output_type', default='matrix', type=str, help="sclar, vector, matrix")
    parser.add_argument('--Normalize', type=bool, default=True)
    parser.add_argument('--GPU', type=int, default=0)

    parser.add_argument('--proportion', default=1.0, type=float)
    parser.add_argument('--val_loss', default=1, type=int, help="1-val_loss, 0-test_loss")
    parser.add_argument('--loss_type', default=1, type=int, help="1-single, 2-multi_man, 3-multi_auto")

    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--test_log', default='test_log', type=str)
    parser.add_argument('--data', default=0, type=int,help="produce data")
    parser.add_argument('--experiment', default=0, type=int,help="Do experiments")
    parser.add_argument('--test', default=0, type=int,help="Do test")
    parser.add_argument('--Times', default=5, type=int, help="the number of experiment")
    return parser

# GEML的data和main都会有的
def GEML_main(parser):
    parser.add_argument('--model', default='GEML', type=str)

    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3, 0, 0], type=list, help="3-6")
    parser.add_argument('--data_names', default="labels-samples_P", type=str)
    parser.add_argument('--output_type', default='matrix', type=str, help="sclar, vector, matrix")
    parser.add_argument('--Normalize', type=bool, default=True)
    parser.add_argument('--proportion', default=1.0, type=float)

    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--val_loss', default=1, type=int, help="1-val_loss, 0-test_loss")
    parser.add_argument('--loss_type', default=1, type=int, help="1-single, 2-multi_man, 3-multi_auto")

    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--test_log', default='test_log', type=str)
    parser.add_argument('--data', default=0, type=int,help="produce data")
    parser.add_argument('--experiment', default=1, type=int,help="Do experiments")
    parser.add_argument('--test', default=0, type=int,help="Do test")
    parser.add_argument('--Times', default=5, type=int, help="the number of experiment")
    return parser

# P的main
def P_main(parser):
    parser.add_argument('--model', default='P', type=str)

    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3, 0, 0], type=list, help="3-6")
    parser.add_argument('--data_names', default="samples_all-samples_P", type=str)
    parser.add_argument('--output_type', default='vector', type=str, help="sclar, vector, matrix")
    parser.add_argument('--Normalize', type=bool, default=True)
    parser.add_argument('--proportion', default=1.0, type=float)

    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--val_loss', default=1, type=int, help="1-val_loss, 0-test_loss")
    parser.add_argument('--loss_type', default=1, type=int, help="1-single, 2-multi_man, 3-multi_auto")

    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--test_log', default='test_log', type=str)
    parser.add_argument('--data', default=True, help="produce data")
    parser.add_argument('--experiment', default=True, help="Do experiments")
    parser.add_argument('--test', default=False, help="Do test")
    parser.add_argument('--Times', default=5, type=int, help="the number of experiment")
    return parser



# dc的main
def dc_main_para():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default='../sz-3/')
    parser.add_argument('--model', default='dc', type=str)
    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    # parser.add_argument('--input_steps', default=[3, 0, 0], type=list, help="3-6")
    parser.add_argument('--input_steps', default=[3,0,0], type=list, help="3-6")
    parser.add_argument('--data_names', default="samples_P-samples_all-samples_after-labels", type=str)
    # parser.add_argument('--data_names', default="samples_P-samples_all-samples_after-labels-samples_D-samples_W", type=str)
    parser.add_argument('--T', default=3, type=str, help="0:dayoftime, 1:dayofweek, 2:both")
    parser.add_argument('--output_type', default='vector', type=str, help="scalar, vector, matrix")
    parser.add_argument('--Normalize', type=bool, default=False)
    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--testMode', type=int, default=2)
    parser.add_argument('--data_log', default='data_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--data_mode', type=str, default="data1", help='data creat mode')
    parser.add_argument('--time_interval', type=int, default=64)
    return parser

# CASCNN的data和main都会有的
def CASCNN_main( parser):
    parser.add_argument('--model', default='CASCNN', type=str)

    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3, 0, 0], type=list, help="3-6")
    parser.add_argument('--data_names', default="labels-samples_P-In-Out",type=str)
    parser.add_argument('--output_type', default='matrix', type=str, help="sclar, vector, matrix")
    parser.add_argument('--Normalize', type=bool, default=True)
    parser.add_argument('--proportion', default=1.0, type=float)

    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--val_loss', default=1, type=int, help="1-val_loss, 0-test_loss")
    parser.add_argument('--loss_type', default=1, type=int, help="1-single, 2-multi_man, 3-multi_auto")

    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--test_log', default='test_log', type=str)
    parser.add_argument('--data', default=0, type=int,help="produce data")
    parser.add_argument('--experiment', default=0, type=int,help="Do experiments")
    parser.add_argument('--test', default=1, type=int,help="Do test")
    parser.add_argument('--Times', default=5, type=int, help="the number of experiment")
    return parser

def CASCNN(parser):
    parser.add_argument('--output_dims', default=[2,1], type=list)
    parser.add_argument('--kernel_size1', default=[3,3], type=list)
    parser.add_argument('--kernel_size2', default=[5,5], type=list)
    return parser


# TR的data和main都会有的
def TR_main( parser):
    parser.add_argument('--model', default='TR', type=str)

    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3, 0, 0], type=list, help="3-6")
    parser.add_argument('--data_names', default="labels-samples_P", type=str)
    parser.add_argument('--output_type', default='matrix', type=str, help="sclar, vector, matrix")
    parser.add_argument('--Normalize', type=bool, default=True)
    parser.add_argument('--proportion', default=1.0, type=float)

    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--val_loss', default=1, type=int, help="1-val_loss, 0-test_loss")
    parser.add_argument('--loss_type', default=1, type=int, help="1-single, 2-multi_man, 3-multi_auto")

    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--test_log', default='test_log', type=str)
    parser.add_argument('--data', default=0, type=int,help="produce data")
    parser.add_argument('--experiment', default=0, type=int,help="Do experiments")
    parser.add_argument('--test', default=0, type=int,help="Do test")
    parser.add_argument('--Times', default=5, type=int, help="the number of experiment")
    return parser

# TRs的data和main都会有的
def TRs_main(parser):
    parser.add_argument('--model', default='TRs', type=str)

    parser.add_argument('--data_type', default='InOD', type=str, help="InOD, OutOD, In, Out")
    parser.add_argument('--input_types', default=['P', 'D', 'W'], type=list)
    parser.add_argument('--input_steps', default=[3, 2, 1], type=list, help="3-6")
    parser.add_argument('--data_names', default="labels-samples_P-samples_D-samples_W", type=str)
    parser.add_argument('--output_type', default='matrix', type=str, help="sclar, vector, matrix")
    parser.add_argument('--Normalize', type=bool, default=True)
    parser.add_argument('--proportion', default=1.0, type=float)

    parser.add_argument('--GPU', type=int, default=0)
    parser.add_argument('--val_loss', default=1, type=int, help="1-val_loss, 0-test_loss")
    parser.add_argument('--loss_type', default=1, type=int, help="1-single, 2-multi_man, 3-multi_auto")

    parser.add_argument('--train_log', default='train_log', type=str)
    parser.add_argument('--result_log', default='result_log', type=str)
    parser.add_argument('--test_log', default='test_log', type=str)
    parser.add_argument('--data', default=0, type=int,help="produce data")
    parser.add_argument('--experiment', default=1, type=int,help="Do experiments")
    parser.add_argument('--test', default=0, type=int,help="Do test")
    parser.add_argument('--Times', default=5, type=int, help="the number of experiment")
    return parser

######################################  模型参数  #################################################
# ANN模型参数
def ANN(parser):
    parser.add_argument('--activations', default=['relu', 'relu'], type=list)
    parser.add_argument('--units', default=[256,256], type=list)
    parser.add_argument('--bn', default=False, type=bool)
    parser.add_argument('--dims', default=2, type=int)
    return parser

# P模型参数
def P(parser):
    parser.add_argument('--activations', default=['relu', 'relu'], type=list)
    parser.add_argument('--units', default=[256, 256], type=list)
    parser.add_argument('--bn', default=False, type=bool)
    parser.add_argument('--dims', default=2, type=int)
    parser.add_argument('--weights', default=[0.5, 0.5], type=list)
    return parser

# LSTM模型参数
def TR(parser):
    parser.add_argument('--intervals', default=32, type=int)#(23-7)*60/30
    parser.add_argument('--heads', default=1, type=int)
    parser.add_argument('--d', default=64, type=int)
    parser.add_argument('--RNN_type', default="GRU", type=str)
    parser.add_argument('--RNN_units', default=256, type=int)
    parser.add_argument('--ANN_units', default=[256], type=list)
    parser.add_argument('--weights', default=[1, 1, 1], type=list)
    return parser

def TRs(parser):
    # parser.add_argument('--units', default=[64, 64], type=list)
    parser.add_argument('--intervals', default=32, type=int)#(23-7)*60/30
    parser.add_argument('--heads', default=1, type=int)
    parser.add_argument('--d', default=64, type=int)
    parser.add_argument('--RNN_type', default="GRU", type=str)
    parser.add_argument('--RNN_units', default=[256], type=list)
    parser.add_argument('--ANN_units', default=[256], type=list)
    parser.add_argument('--weights', default=[1, 1, 1], type=list) #没用到
    parser.add_argument('--is_TE', default=1, type=int)
    parser.add_argument('--is_SE', default=1, type=int)
    parser.add_argument('--is_RNN', default=1, type=int)
    return parser

# RNNs模型参数
def RNNs(parser):
    parser.add_argument('--units', default=[256,256], type=list)
    parser.add_argument('--RNN_Type', default='LSTM', type=str)
    return parser

# RNNs模型参数
def ConvLSTM(parser):
    # parser.add_argument('--units', default=[256,256], type=list)
    # parser.add_argument('--RNN_Type', default='LSTM', type=str)
    return parser

# GCN模型参数
def GCN(parser):
    parser.add_argument('--GCN_activations', default=["relu", "relu"], type=list)
    parser.add_argument('--GCN_units', default=[256,256], type=list)
    parser.add_argument('--Ks', default=[None,None], type=list)
    parser.add_argument('--weights', default=[0.5, 0.25, 0.25], type=list)
    return parser

# GEML模型参数
def GEML(parser):
    parser.add_argument('--GCN_activations', default=["sigmoid", "sigmoid"], type=list)
    parser.add_argument('--RNN_units', default=[128], type=list)
    parser.add_argument('--GCN_units', default=[128,128], type=list)
    parser.add_argument('--Ks', default=[None,None], type=list)
    parser.add_argument('--RNN_type', default="LSTM", type=str)
    parser.add_argument('--weights', default=[0.5, 0.25, 0.25], type=list)
    return parser
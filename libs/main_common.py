# conding=utf-8
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import pandas as pd
import os
import numpy as np
import time
from sklearn import linear_model
from libs import utils, metrics, data_common
import datetime
from trmf import Model, train
import tqdm


def main(args):
    for i in range(args.Times):
        # 产生数据
        if args.data:
            data_common.data_process(args)
        # 实验
        if args.experiment:
            experiment(args)
            tf.reset_default_graph()
    # 是否测试
    if args.test:
        test_model(args)


######################################## Regression#####################################
def HA_evaluate(tests):
    samples = tests[1]
    preds = np.mean(samples, axis=1)  # (M,D,N,N) => (M,N,N)
    return preds


def Ridge_evaluate(trains, tests):
    # 模型训练 alphas=[1,0.1,0.01,0.001]
    clf = linear_model.RidgeCV(cv=5).fit(trains[1], trains[0])
    preds = clf.predict(tests[1])
    return preds


def Lasso_evaluate(trains, tests):
    # 模型训练 alphas=[1,0.1,0.01,0.001]
    clf = linear_model.MultiTaskLassoCV(cv=5, max_iter=2000, random_state=0).fit(trains[1], trains[0])
    preds = clf.predict(tests[1])
    return preds


def Mat_evaluate(tests):
    data = tests[1]
    # data = trains
    shape = data.shape
    # lag_set = np.array(list(range(1, 65)), dtype=np.uint32)
    lag_set = np.array(list(range(1, 65)), dtype=np.uint32)
    # lag_set = np.array(list(range(0+1, 32+1)), dtype=np.uint32)
    print("lag_set: ",np.min(lag_set)," to ",np.max(lag_set))
    k = 40
    lambdaI = 2
    lambdaAR = 625
    lambdaLag = 0.5
    # window_size = 24
    window_size = 1
    nr_windows = 7
    max_iter = 400
    threshold = 0
    threads = 40
    seed = 0
    missing = False
    transform = True
    # threshold=None
    verbose = 0
    # print(Y)
    pred = []
    premod=None
    for i in tqdm.trange(shape[0]):
        curr_model = Model.initialize(data[i, ...], lag_set, k, seed=seed, warm_start_model=None, transform=transform)
        curr_model = train(data[i, :, :], curr_model,
                           lambdaI=lambdaI, lambdaAR=lambdaAR, lambdaLag=lambdaLag,
                           max_iter=max_iter, missing=missing, threads=threads, verbose=verbose)
        res, _ = curr_model.forecast(1, Ynew=None, threshold=threshold)
        # premod=curr_model
        # print(res)
        # print(type(res))
        pred.append(res)
    return np.array(pred)


def regression_main(args):
    log_dir, save_dir, data_dir, dir_name = utils.create_dir_PDW(args)  # 按输入模式，产生文件夹
    log_res = utils.create_log(args, args.result_log)  # 结果log
    log = utils.create_log(args, args.train_log)  # 训练log
    utils.log_string(log, str(args)[10: -1])  # 打印超参数
    # 装载数据:X=(M,T,N,N), Y=(M,N,N)
    data = data_common.load(args, log, dir_name)
    Time = data[0][..., -3:]
    data[0], data[1] = data[0][..., :-3], data[1][..., :-3]  # 回归模型不需要时间特征

    print("Time shape", Time.shape)

    if args.model in ["Mat"]:
        print("This is data process")
        pass

    if args.model in ["Ridge", "Lasso"]:
        # 数据变形:X=(M,T,N,N)=>(MN,TN), Y=(M,N,N)=>(MN,N)
        # data = data_common.Regression_data_vector(data) # 预测向量,向量似乎无法预测
        data = data_common.Regression_data_scalar(data)  # 预测标量
        # data = data_common.Regression_random_data()  # 随机数据验证

    for j in range(args.Times):
        utils.log_string(log, "experiment:" + str(j))
        # 打乱数据集
        data = data_common.shuffle(data)
        # 取部分数据
        Data = data_common.proportion(log, data, args.proportion)
        # 划分训练集/验证集/测试集
        trains, vals, tests = data_common.data_split(args, log, Data)
        if args.model == "HA":
            preds = HA_evaluate(tests)
        elif args.model == "Ridge":
            preds = Ridge_evaluate(trains, tests)
        elif args.model == "Lasso":
            preds = Lasso_evaluate(trains, tests)
        elif args.model == "Mat":
            print("This is prediction")
            tests[0] = data[0][j*10:(j+1)*10]
            tests[1] = data[1][j * 10:(j + 1) * 10]
            preds = Mat_evaluate(tests)

        preds = np.round(preds).astype(np.float32)
        print("min", np.min(preds))
        labels = tests[0]
        mae, rmse, wmape, smape = metrics.calculate_metrics(preds, labels, null_val=args.Null_Val)
        message = "MAE\t%.4f\tRMSE\t%.4f\tWMAPE\t%.4f\tSMAPE\t%.4f" % (mae, rmse, wmape, smape)
        utils.log_string(log_res, message)
        utils.log_string(log, message)

    if args.test:

        mat = np.load("{}original/matrix_in_30.npz".format(args.dataset_dir))["matrix"]
        mat = np.sum(mat, axis=-1)
        mat_2 = np.zeros_like(mat)
        mat_2 = mat_2 - 1
        M, N, _ = Time.shape
        trains, vals, tests = data_common.data_split(args, log, data)

        if args.model == "HA":
            labels = data[0]
            preds = HA_evaluate(data)
        elif args.model == "Ridge":
            preds = Ridge_evaluate(trains, data)
            preds = np.reshape(preds, [M, N, N])
            labels = np.reshape(data[0], [M, N, N])
        elif args.model == "Lasso":
            preds = Lasso_evaluate(trains, data)
            preds = np.reshape(preds, [M, N, N])
            labels = np.reshape(data[0], [M, N, N])
        preds = np.round(preds).astype(np.float32)
        print(preds.shape)
        print(labels.shape)

        for i in range(M):
            for j in range(N):
                day = int(Time[i][j][2])
                T = int(Time[i][j][0])
                for k in range(N):

                    if labels[i, j, k] == mat[day, T, j, k]:
                        mat_2[day, T, j, k] = preds[i, j, k]

                    else:
                        print(day, T, k)
                        print(labels[i, j, k], mat[day, T, j, k])
                        raise ValueError
        np.savez_compressed(os.path.join(data_dir, "pre_mat.npz"), matrix=mat_2)

    utils.log_string(log, 'input_type:%s_%s, prediction type:%s' % (args.data_type, dir_name, args.output_type))
    utils.log_string(log, 'Finish\n')


########################################  Nerual Network #####################################################

def test_model(args):
    log_dir, save_dir, data_dir, dir_name = utils.create_dir_PDW(args)  # 按输入模式，产生文件夹

    log_res = utils.create_log(args, args.test_log)  # 结果log
    log = utils.create_log(args, args.train_log)  # 训练log
    utils.log_string(log, str(args)[10: -1])  # 打印超参数
    model_file = save_dir + '/'

    data, Time = data_common.test_data_process(args, log, dir_name, data_dir)

    # 配置GPU
    sess = GPU(number=args.GPU)

    ckpt = tf.train.get_checkpoint_state(model_file)
    print(ckpt)
    if ckpt and ckpt.model_checkpoint_path:
        saver2 = tf.train.import_meta_graph(ckpt.model_checkpoint_path + ".meta")
        saver2.restore(sess, tf.train.latest_checkpoint(model_file))
    else:
        raise ValueError

    graph = tf.get_default_graph()  # 获取当前默认计算图

    if args.model in ["ANN", "RNNs", "GCN", "TR", "ConvLSTM"]:
        samples = graph.get_tensor_by_name("samples:0")
        labels = graph.get_tensor_by_name("lables:0")
        placeholders = [labels, samples]

    if args.model == "TRs":
        samples_P = graph.get_tensor_by_name("samples_P:0")
        samples_D = graph.get_tensor_by_name("samples_D:0")
        samples_W = graph.get_tensor_by_name("samples_W:0")
        labels = graph.get_tensor_by_name("lables:0")
        placeholders = [labels, samples_P, samples_D, samples_W]

    if args.model == "Com":
        samples_P_before_C = graph.get_tensor_by_name("samples_P_before_C:0")
        samples_P_after_W_C = graph.get_tensor_by_name("samples_P_after_W_C:0")
        samples_P_before_odt_C = graph.get_tensor_by_name("samples_P_before_odt_C:0")
        labels = graph.get_tensor_by_name("labels:0")
        placeholders = [labels, samples_P_before_C, samples_P_before_odt_C, samples_P_after_W_C]


    if args.model == "GEML":
        samples = graph.get_tensor_by_name("samples:0")
        graph_sem = graph.get_tensor_by_name("graph_sem:0")
        labels = graph.get_tensor_by_name("labels:0")
        placeholders = [labels, samples, graph_sem]

    if args.model == "Com":
        labels = graph.get_tensor_by_name("labels:0")
        samples_P_before_C = graph.get_tensor_by_name("samples_P_before_C:0")
        samples_P_after_W_C = graph.get_tensor_by_name("samples_P_after_W_C:0")
        samples_P_before_odt_C = graph.get_tensor_by_name("samples_P_before_odt_C:0")
        IN = graph.get_tensor_by_name("IN:0")
        placeholders = [labels, samples_P_before_C, samples_P_after_W_C, samples_P_before_odt_C, IN]


    if args.model == "CASCNN":
        labels = graph.get_tensor_by_name("lables:0")
        samples = graph.get_tensor_by_name("samples:0")
        in_flow = graph.get_tensor_by_name("in_flow:0")
        out_flow = graph.get_tensor_by_name("out_flow:0")
        placeholders = [labels, samples, in_flow, out_flow]


    preds = graph.get_tensor_by_name("preds:0")

    preds_all = []
    test_batch = 100  # 要求大于等于2
    M = data[0].shape[0]
    for data_index in range(int((M - 1) / test_batch) + 1):
        # print("data_index = ",data_index)
        feed_dict = {}
        if (data_index + 1) * test_batch > M:
            for i in range(len(placeholders)):  # 生成 (1,N,N)的数据,解决OOM问题.为了加快速度可以增大,但是需要考虑对齐
                feed_dict[placeholders[i]] = data[i][data_index * test_batch:M, ...]
        else:
            for i in range(len(placeholders)):  # 生成 (1,N,N)的数据,解决OOM问题.为了加快速度可以增大,但是需要考虑对齐
                feed_dict[placeholders[i]] = data[i][data_index * test_batch:(data_index + 1) * test_batch, ...]

        tmp = sess.run(preds, feed_dict=feed_dict)
        if data_index == 0:
            preds_all = tmp
        else:
            preds_all = np.concatenate((preds_all, tmp), axis=0)

    labels_all = data[0]
    preds_all = np.round(preds_all).astype(np.float32)

    if args.model == "Com":
        # print("COM")
        path = os.path.join(args.dataset_dir, 'all_models', args.data_mode, dir_name)  # 存放的文件夹路径
        path = os.path.join(path, "samples_CP_InOD.npz")

        # print(path,Time.shape)
        # print(preds_all.shape)
        np.savez_compressed(path, np.concatenate((preds_all, Time), axis=-1))  # 将时间维度加上
    else:
        mat = np.load("{}original/matrix_in_30.npz".format(args.dataset_dir))["matrix"]
        mat = np.sum(mat, axis=-1)
        mat_2 = np.zeros_like(mat)
        mat_2 = mat_2 - 1

        print(preds_all.shape)
        print(Time.shape)
        print(preds_all.shape)

        if args.model in ["RNNs"]:
            preds_res = data_common.RNNs_data_vector_inv(preds_all)
            labels = data_common.RNNs_data_vector_inv(labels_all)
        else:
            preds_res = preds_all
            labels = labels_all

        M, N, _ = Time.shape
        for i in range(M):
            for j in range(N):
                day = int(Time[i][j][2])
                T = int(Time[i][j][0])
                for k in range(N):
                    # mat_2[day][T][j][k] = preds_all[i][j][k]
                    if labels[i, j, k] == mat[day, T, j, k]:
                        mat_2[day, T, j, k] = preds_res[i, j, k]
                        pass
                    else:
                        print(labels[i, j, k], mat[labels[i, j, N + 1], labels[i, j, N], j, k])
                        raise ValueError

        np.savez_compressed(os.path.join(data_dir, "pre_mat.npz"), matrix=mat_2)

    mae, rmse, wmape, smape = metrics.calculate_metrics(preds_all, labels_all, null_val=args.Null_Val)
    Message = "MAE\t%.4f\tRMSE\t%.4f\tWMAPE\t%.4f\tSMAPE\t%.4f" % (mae, rmse, wmape, smape)
    utils.log_string(log, Message)

    utils.log_string(log, 'Finish\n')

    utils.log_string(log_res, Message)
    sess.close()


def get_feed_dicts(data, placeholders, shuffle=0):
    num_batch = data['arr_0'].shape[0]

    feed_dicts = []
    if shuffle:
        per = list(np.random.permutation(num_batch))  # 随机划分
    else:
        per = range(num_batch)

    for j in per:
        feed_dict = {}
        for i in range(len(placeholders)):
            feed_dict[placeholders[i]] = data['arr_%s' % i][j, ...]
        feed_dicts.append(feed_dict)
    return feed_dicts


def choose_model(args):
    if args.model == "ANN":
        import model_ANN as model
    elif args.model == "RNNs":
        import model_RNNs as model
    elif args.model == "ConvLSTM":
        import model_ConvLSTM as model
    elif args.model == "GCN":
        import model_GCN as model
    elif args.model == "GEML":
        import model_GEML as model
    elif args.model == "TR":
        import model_TR as model
    elif args.model == "TRs":
        import model_TRs as model
    elif args.model == 'P':
        import model_P as model
    elif args.model == 'ConvLSTM':
        import model_ConvLSTM as model
    elif args.model == 'Com':
        import model_complete as model
    elif args.model == "CASCNN":
        import model_CASCNN as model
    else:
        raise ValueError
    return model


def experiment(args):
    print(args)

    model = choose_model(args)  # 选择模型
    log_dir, save_dir, data_dir, dir_name = utils.create_dir_PDW(args)  # 按输入模式，产生文件夹
    log_res = utils.create_log(args, args.result_log)  # 结果log
    log = utils.create_log(args, args.train_log)  # 训练log
    utils.log_string(log, str(args)[10: -1])  # 打印超参数
    model_file = save_dir + '/'

    # 装载数据
    data, mean, std = data_common.load_data(args, log, data_dir, yes=args.Normalize)

    if args.model == "GCN":
        path = os.path.join(data_dir, args.graph_name)
        graph = np.load(path)['graph']  # (N,N)
    elif args.model in ["TR", "TRs"]:
        path = os.path.join(args.dataset_dir, 'original', 'SE.npz')
        SE = np.load(path)["SE"].astype(np.float32)
        utils.log_string(log, 'SE:  {}'.format(SE.shape))
    elif args.model == "GEML":
        path = os.path.join(data_dir, 'graph_geo.npz')
        graph_geo = np.load(path)['graph'].astype(np.float32)

    utils.log_string(log, 'Data Loaded Finish...')

    # 模型编译
    utils.log_string(log, 'compiling model...')
    P, N = args.P, args.N
    F_in, F_out = data[1]['arr_1'].shape[-1], N
    utils.log_string(log, "P-%s, F_in-%s, F_out-%s" % (P, F_in, F_out))

    if args.model == "ANN":
        # (V,B,N,PN)=>(V,B,N,N)
        placeholders = model.placeholder_vector(N, F_in, F_out)
        labels, samples = placeholders
        preds, In_preds, Out_preds = model.Model(args, mean, std, samples, F_out)
    if args.model == "RNNs":
        # (VN,B,T,N)=>(VN,B,N)
        placeholders = model.placeholder(P, F_in, F_out)
        labels, samples = placeholders
        preds, In_preds, Out_preds = model.Model(args, mean, std, samples, F_out)
    if args.model == "ConvLSTM":
        print(data[1]['arr_1'].shape)
        batch_size = data[1]['arr_1'].shape[1]
        placeholders = model.placeholder(batch_size, P, F_in, F_out)
        labels, samples = placeholders
        preds, In_preds, Out_preds = model.Model(mean, std, samples, N)
    if args.model == "GCN":
        placeholders = model.placeholder(N, F_in, F_out)
        labels, samples = placeholders
        preds, In_preds, Out_preds = model.Model(args, mean, std, samples, graph, F_out)
    if args.model == "TR":
        placeholders = model.placeholder(P, N, F_in, F_out)
        labels, samples = placeholders
        preds, In_preds, Out_preds = model.Model(args, mean, std, samples, SE)
    if args.model == "TRs":
        D, W = data[1]['arr_2'].shape[-3], data[2]['arr_3'].shape[-3]
        placeholders = model.placeholder(P, D, W, N, F_in, F_out)
        labels, samples_P, samples_D, samples_W = placeholders
        samples_P = tf.identity(samples_P, name='samples_P')
        samples_D = tf.identity(samples_D, name='samples_D')
        samples_W = tf.identity(samples_W, name='samples_W')
        samples = [samples_P, samples_D, samples_W]
        preds, In_preds, Out_preds = model.Model(args, mean, std, samples, SE, N)
    if args.model == "GEML":
        placeholders = model.placeholder(P, N, F_in, F_out)
        labels, samples, graph_sem = placeholders
        # graph_sem = tf.identity(graph_sem, name='graph_sem')
        preds, In_preds, Out_preds = model.Model(args, mean, std, samples, graph_geo, graph_sem)

    if args.model == "P":
        placeholders = model.placeholder(P, N, F_in, F_out)
        labels, samples = placeholders
        preds, In_preds, Out_preds = model.Model(args, mean, std, samples, F_out)

    if args.model == "Com":
        placeholders = model.placeholder(P, N, F_in, F_out, data[0]['arr_3'].shape[-1])
        labels, samples_P_before_C, samples_P_after_W_C, samples_P_before_odtR1_C,samples_P_before_odtD1_C, IN = placeholders
        preds, In_preds, Out_preds = model.Model(samples_P_before_C, samples_P_after_W_C, samples_P_before_odtR1_C,
                                                 samples_P_before_odtD1_C, IN,
                                                 std, mean, F_out)

        # labels, samples_P_before_C, samples_P_after_W_C, samples_P_before_odt_C, IN = placeholders
        # preds, In_preds, Out_preds = model.Model(samples_P_before_C, samples_P_after_W_C, samples_P_before_odt_C, IN, std, mean, F_out)
        # preds, In_preds, Out_preds = model.Model(samples_P_before_C, samples_P_after_W_C, samples_P_before_odt_C,
        #                                           IN,std, mean, F_out)


    if args.model == "CASCNN":
        placeholders = model.placeholder_vector(P, N)
        labels, samples, in_flow, out_flow = placeholders
        # preds, In_preds, Out_preds = model.Model(samples_P_before_C, samples_P_after_W_C, samples_P_before_odt_C, IN, std, mean, F_out)
        preds, In_preds, Out_preds = model.Model(args, std, mean, samples, in_flow, out_flow)

    # 改名字
    preds = tf.identity(preds, name='preds')
    # labels = tf.identity(labels, name='common_labels')
    # samples = tf.identity(samples, name='samples')
    # print(preds)
    # print(labels)
    # print(samples)

    Preds = [preds, In_preds, Out_preds]
    In_labels, Out_labels = tf.reduce_sum(labels, axis=-1), tf.reduce_sum(labels, axis=-2)
    Labels = [labels, In_labels, Out_labels]

    # sys.exit()
    if args.loss_type == 1:  # 单个目标
        loss = metrics.masked_mse_tf(preds, labels, null_val=args.Null_Val)  # 损失
    elif args.loss_type == 2:  # 多个目标-自动权重
        loss = metrics.total_loss(args, Preds, Labels)
    elif args.loss_type == 3:  # 多个目标-人为权重
        loss = metrics.total_loss_W(args, Preds, Labels)

    lr, new_lr, lr_update, train_op = optimization(args, loss)  # 优化
    utils.print_parameters(log)  # 打印参数
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)  # 保存模型

    # 配置GPU
    sess = GPU(number=args.GPU)

    if args.continue_train:
        val_loss_min, Epoch = restore(log, sess, model_file, saver)
    else:
        # 初始化模型
        val_loss_min = np.inf
        Epoch = 0
        sess.run(tf.global_variables_initializer())
    wait = 0
    step = 0

    utils.log_string(log, "initializer successfully")

    utils.log_string(log, '**** training model ****')

    Message = ''
    save_loss = [[], [], []]
    epoch = Epoch
    while (epoch < args.max_epoch):
        # for epoch in range(Epoch, args.max_epoch):
        # 降低学习率
        if wait >= args.patience:
            val_loss_min, epoch = restore(log, sess, model_file, saver)
            step += 1
            wait = 0
            New_Lr = max(args.min_learning_rate, args.base_lr * (args.lr_decay_ratio ** step))
            sess.run(lr_update, feed_dict={new_lr: New_Lr})
            # 删除多余的loss
            if epoch > args.patience:
                for k in range(len(save_loss)):
                    save_loss[k] = save_loss[k][:-args.patience]
            if step > args.steps:
                utils.log_string(log, 'early stop at epoch: %04d' % (epoch))
                break

        # 打印当前时间/训练轮数/lr
        utils.log_string(log,
                         '%s | epoch: %04d/%d, lr: %.4f' %
                         (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, args.max_epoch, sess.run(lr)))

        # 计算训练集/验证集/测试集损失
        types = ['train', 'val', 'test']
        results = []
        for i in range(len(types)):
            feed_dicts = get_feed_dicts(data[i], placeholders, args.train_shuffle)
            result, message = caculation(args, log, sess, data_dir, feed_dicts, labels, preds, train_op, loss,
                                         type=types[i])
            if i == 2:  # 获取测试集的性能指标
                message_cur = message
            results.append(result)
        message = "loss=> train:%.4f val:%.4f test:%.4f time=> train:%.1f val:%.1f test:%.1f" % (
        results[0][0], results[1][0], results[2][0], results[0][1], results[1][1], results[2][1])
        utils.log_string(log, message)

        # 存储损失
        save_loss[0].append(results[0][0])
        save_loss[1].append(results[1][0])
        save_loss[2].append(results[2][0])

        # 更新最小损失
        if args.val_loss == 1:
            val_loss = results[1][0]
        else:
            val_loss = results[2][0]
        wait, val_loss_min, Message = update_val_loss(log, sess, saver, model_file, epoch, wait, val_loss, val_loss_min,
                                                      message_cur, Message)
        epoch += 1
    print("Message", Message)
    # 存储好损失
    path = os.path.join(data_dir, 'losses.npz')
    np.savez_compressed(path, np.array(save_loss))
    utils.log_string(log_res, Message)
    sess.close()


'''
配置优化器/学习率/剪裁梯度/反向更新/
'''


def optimization(args, loss):
    lr = tf.Variable(tf.constant_initializer(args.base_lr)(shape=[]),
                     dtype=tf.float32, trainable=False, name='learning_rate')  # (F, F1)

    # lr = tf.get_variable('learning_rate', initializer=tf.constant(args.base_lr), trainable=False)
    new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
    lr_update = tf.assign(lr, new_lr)
    if args.opt == 'adam':
        optimizer = tf.train.AdamOptimizer(lr, epsilon=1e-3)
    elif args.opt == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    elif args.opt == 'amsgrad':
        optimizer = tf.train.AMSGrad(lr, epsilon=1e-3)

    # clip
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    grads, _ = tf.clip_by_global_norm(grads, args.max_grad_norm)
    global_step = tf.train.get_or_create_global_step()
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')
    return lr, new_lr, lr_update, train_op


'''
配置GPU
'''


def GPU(number):
    # GPU configuration
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(number)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


'''
恢复模型或初始化模型
'''


def restore(log, sess, model_file, saver):
    ckpt = tf.train.get_checkpoint_state(model_file)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        Epoch = int(ckpt.model_checkpoint_path.split('-')[-1]) + 1
        val_loss_min = np.load(model_file + 'val_loss_min.npz')['loss']
        message = "restore successfully, path:%s, Epoch:%d" % (ckpt.model_checkpoint_path, Epoch)
        utils.log_string(log, message)
    else:
        val_loss_min = np.inf
        Epoch = 0
        sess.run(tf.global_variables_initializer())
        utils.log_string(log, "initializer successfully")
    return val_loss_min, Epoch


'''
更新损失最小值，存储最好的模型，存储最小损失，npz，'loss'
'''


def update_val_loss(log, sess, saver, model_file, epoch, wait, loss, val_loss_min, message_cur, Message):
    # choose best test_loss
    if loss < val_loss_min:
        wait = 0
        val_loss_min = loss
        saver.save(sess, model_file, epoch)
        Message = message_cur
        np.savez(model_file + 'val_loss_min.npz', loss=val_loss_min)
        utils.log_string(log, "save %02d" % epoch)

        # graph = tf.get_default_graph()  # 获取当前默认计算图
        # w0 = graph.get_tensor_by_name("weight:0")
        # print(sess.run(w0))
        # w00 = graph.get_tensor_by_name("weight0:0")
        # print(sess.run(w00))
        # w01 = graph.get_tensor_by_name("weight1:0")
        # print(sess.run(w01))
        # w02 = graph.get_tensor_by_name("weight2:0")
        # print(sess.run(w02))

    else:
        wait += 1
    return wait, val_loss_min, Message


'''
查看预测的结果
'''


def pred_label(log, test_pred, test_label, data_dir):
    pred = np.reshape(test_pred, [-1, test_pred.shape[-1]])
    label = np.reshape(test_label, [-1, test_label.shape[-1]])
    diff = pred - label
    save(diff, data_dir, name='diff')
    save(pred, data_dir, name='pred')
    save(label, data_dir, name='label')
    utils.log_string(log, "save succesfully")


def save(values, data_dir, name):
    path = os.path.join(data_dir, '%s.xlsx' % name)
    df = pd.DataFrame(values)
    df.to_excel(path)


'''
计算一个epoch，训练时反向更新，测试时进行预测
'''


def caculation(args, log, sess, data_dir, feed_dicts, labels, preds, train_op, loss, type="train"):
    start = time.time()
    loss_all = []
    preds_all = []
    labels_all = []
    message_res = ''
    for feed_dict in feed_dicts:
        if type == "train":
            sess.run([train_op], feed_dict=feed_dict)
            # loss_list = sess.run([Losses], feed_dict=feed_dict)
        batch_loss = sess.run([loss], feed_dict=feed_dict)
        loss_all.append(batch_loss)
        if type == "test":
            batch_labels, batch_preds = sess.run([labels, preds], feed_dict=feed_dict)
            preds_all.append(batch_preds)
            labels_all.append(batch_labels)
    loss_mean = np.mean(loss_all)
    Time = time.time() - start

    if type == "test":
        preds_all = np.stack(preds_all, axis=0)
        preds_all = np.round(preds_all).astype(np.float32)
        labels_all = np.stack(labels_all, axis=0)
        mae, rmse, wmape, smape = metrics.calculate_metrics(preds_all, labels_all, null_val=args.Null_Val)
        message = "Test=> MAE:{:.4f} RMSE:{:.4f} WMAPE:{:.4f} SMAPE:{:.4f}".format(mae, rmse, wmape, smape)
        message_res = "MAE\t%.4f\tRMSE\t%.4f\tWMAPE\t%.4f\tSMAPE\t%.4f" % (mae, rmse, wmape, smape)
        utils.log_string(log, message)
        # 查看预测效果
        # pred_label(log, preds_all, labels_all, data_dir)
        # sys.exit()
    return [loss_mean, Time], message_res


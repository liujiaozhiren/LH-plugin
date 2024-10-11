import logging
import os, sys
import pickle
import time
from typing import List
from datetime import datetime, timezone, timedelta
import numpy as np
import torch

import tools.config as config
import tools.sampling_methods as sm
import tools.test_methods as tm
from geo_rnns.neutraj_model import NeuTraj_Network
from geo_rnns.wrloss import WeightedRankingLoss
from tqdm import tqdm

from lorentz.handler import LH_trainer, EmbeddingModelHandler

parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

sys.path.append(parent_parent_dir)

from lorentz.transfer import cal_top10_acc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU


def pad_sequence(traj_grids: List[List[List]], maxlen=100, pad_value=0.0):
    """
    padding
    Args:
        traj_grids: 原始轨迹
        maxlen: 最大长度
        pad_value: 空值

    Returns:

    """
    paddec_seqs = []
    for traj in traj_grids:
        pad_r = np.zeros_like(traj[0]) * pad_value
        while (len(traj) < maxlen):
            traj.append(pad_r)  # 在末尾不上同纬度的pad_value填充的矩阵
        paddec_seqs.append(traj)
    return paddec_seqs


class NeuTrajTrainer(object):
    def __init__(self, tagset_size,
                 batch_size, sampling_num, learning_rate=config.learning_rate):
        """

        Args:
            tagset_size: 目标embedding的长度
            batch_size: batch大小
            sampling_num:
            learning_rate: 学习率
        """
        self.target_size = tagset_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num
        self.learning_rate = learning_rate

    def data_prepare(self, griddatapath=config.gridxypath,
                     coordatapath=config.corrdatapath,
                     distancepath=config.distancepath,
                     train_radio=config.seeds_radio):
        """

        Args:
            griddatapath: (裁剪后的grid id序列集合,[],最长经过cell去重的坐标轨迹长度)的pickle文件
            coordatapath: (经过cell去重的坐标轨迹集合,[],最长经过cell去重的坐标轨迹长度)的pickle文件
            distancepath: (trans_len,轨迹总数)的距离矩阵, 在toy数据集里是(1800,1874)
            train_radio: 选取的种子轨迹的比例

        Returns:

        """
        dataset_length = config.datalength  # 数据集长度
        traj_grids, useful_grids, max_len = pickle.load(
            open(griddatapath, 'rb'))

        # 存储每个网格序列的长度
        self.trajs_length = [len(j) for j in traj_grids][:dataset_length]
        # 网格大小
        self.grid_size = config.gird_size
        # 最长经过cell去重的坐标轨迹长度
        self.max_length = max_len

        # 组成一个List[List[List[x+spatial_width,y+spatial_width]]], 长度为dataset_length,也就是1800个
        grid_trajs = [[[i[0] + config.spatial_width, i[1] + config.spatial_width] for i in tg]
                      for tg in traj_grids[:dataset_length]]

        # (经过cell去重的坐标轨迹集合,[],最长经过cell去重的坐标轨迹长度)
        traj_grids, useful_grids, max_len = pickle.load(
            open(coordatapath, 'rb'))

        # 求 lat,lon的均值和标准差
        # r:(lat,lon)
        x, y = [], []
        for traj in traj_grids:
            for r in traj:
                x.append(r[0])
                y.append(r[1])
        meanx, meany, stdx, stdy = np.mean(x), np.mean(y), np.std(x), np.std(y)
        # 对坐标轨迹进行标准化(归一化)
        traj_grids = [[[(r[0] - meanx) / stdx, (r[1] - meany) / stdy]
                       for r in t] for t in traj_grids]
        # 然后去前dataset_length个
        coor_trajs = traj_grids[:dataset_length]

        # 参与训练的网格轨迹数量
        train_size = int(len(grid_trajs) * train_radio /
                         self.batch_size) * self.batch_size
        # print(f"{train_size=}")

        grid_train_seqs = grid_trajs[:train_size]
        grid_test_seqs = grid_trajs[train_size:]
        coor_train_seqs = coor_trajs[:train_size]
        coor_test_seqs = coor_trajs[train_size:]

        self.grid_trajs = grid_trajs  # 存一下所有的网格轨迹
        self.grid_train_seqs = grid_train_seqs  # 存一下网格轨迹训练集
        self.coor_trajs = coor_trajs  # 存一下所有的cell去重坐标轨迹
        self.coor_train_seqs = coor_train_seqs  # 存一下cell去重坐标轨迹训练集

        # 对数据进行了扩充, 将原本的网格轨迹和cell去重的坐标轨迹合并到一起了
        pad_trjs = []
        for i, t in enumerate(grid_trajs):
            traj = []
            for j, p in enumerate(t):
                traj.append(
                    [coor_trajs[i][j][0], coor_trajs[i][j][1], p[0], p[1]])
            pad_trjs.append(traj)

        #print(f"Padded Trajs shape:{len(pad_trjs)}")

        self.train_seqs: List[List[List[float, float, int, int]]] = pad_trjs[:train_size]  # 保存前train_size个合并后的轨迹
        self.padded_trajs = np.array(pad_sequence(pad_trjs, maxlen=max_len))  # 对合并后的轨迹进行padding, 默认空值是0.0

        distance = pickle.load(open(distancepath, 'rb'))  # 距离矩阵(1800,1874)
        distance = np.array(distance)

        max_dis = distance.max()  # 最大的轨迹间距离
        #print(f'max value in distance matrix :{max_dis}')

        #print(config.distance_type)
        if config.distance_type == 'dtw' or config.distance_type == 'hausdorff' or config.distance_type == 'sspd' or config.distance_type == 'erp':
            distance = distance / max_dis

        #print(f"Distance shape:{distance[:train_size].shape}")
        train_distance = distance[:train_size, :train_size]  # 取出有用的那部分

        #print(f"Train Distance shape:{train_distance.shape}")
        self.distance = distance  # 保存距离矩阵
        self.train_distance = train_distance  # 保存有用的距离矩阵

    def batch_generator(self, train_seqs, train_distance):
        j = 0
        while j < len(train_seqs):
            anchor_input, trajs_input, negative_input, distance, negative_distance = [], [], [], [], []
            anchor_input_len, trajs_input_len, negative_input_len = [], [], []
            anchors_idx, trajs_idx, negs_idx = [], [], []
            batch_trajs_keys = {}
            batch_trajs_input, batch_trajs_len = [], []
            for i in range(self.batch_size):
                # sampling_index_list = sm.random_sampling(len(self.train_seqs),j+i)
                sampling_index_list = sm.distance_sampling(
                    self.distance, len(self.train_seqs), j + i)
                negative_sampling_index_list = sm.negative_distance_sampling(
                    self.distance, len(self.train_seqs), j + i)

                trajs_input.append(train_seqs[j + i])
                anchor_input.append(train_seqs[j + i])
                negative_input.append(train_seqs[j + i])
                trajs_idx.append(j + i)
                anchors_idx.append(j + i)
                negs_idx.append(j + i)
                if j + i not in batch_trajs_keys:
                    batch_trajs_keys[j + i] = 0
                    batch_trajs_input.append(train_seqs[j + i])
                    batch_trajs_len.append(self.trajs_length[j + i])

                anchor_input_len.append(self.trajs_length[j + i])
                trajs_input_len.append(self.trajs_length[j + i])
                negative_input_len.append(self.trajs_length[j + i])

                distance.append(1)
                negative_distance.append(1)

                for traj_index in sampling_index_list:
                    anchor_input.append(train_seqs[j + i])
                    trajs_input.append(train_seqs[traj_index])
                    anchors_idx.append(j + i)
                    trajs_idx.append(traj_index)

                    anchor_input_len.append(self.trajs_length[j + i])
                    trajs_input_len.append(self.trajs_length[traj_index])

                    if traj_index not in batch_trajs_keys:
                        batch_trajs_keys[j + i] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])

                    distance.append(
                        np.exp(-float(train_distance[j + i][traj_index]) * config.mail_pre_degree))

                for traj_index in negative_sampling_index_list:
                    negative_input.append(train_seqs[traj_index])
                    negs_idx.append(traj_index)
                    negative_input_len.append(self.trajs_length[traj_index])
                    negative_distance.append(
                        np.exp(-float(train_distance[j + i][traj_index]) * config.mail_pre_degree))

                    if traj_index not in batch_trajs_keys:
                        batch_trajs_keys[j + i] = 0
                        batch_trajs_input.append(train_seqs[traj_index])
                        batch_trajs_len.append(self.trajs_length[traj_index])
            # normlize distance
            # distance = np.array(distance)
            # distance = (distance-np.mean(distance))/np.std(distance)
            max_anchor_length = max(anchor_input_len)
            max_sample_lenght = max(trajs_input_len)
            max_neg_lenght = max(negative_input_len)
            anchor_input = pad_sequence(anchor_input, maxlen=max_anchor_length)
            trajs_input = pad_sequence(trajs_input, maxlen=max_sample_lenght)
            negative_input = pad_sequence(
                negative_input, maxlen=max_neg_lenght)
            batch_trajs_input = pad_sequence(batch_trajs_input, maxlen=max(max_anchor_length, max_sample_lenght,
                                                                           max_neg_lenght))

            yield (
                [np.array(anchor_input), np.array(trajs_input), np.array(negative_input), np.array(batch_trajs_input), (anchors_idx, trajs_idx, negs_idx)],
                [anchor_input_len, trajs_input_len,
                 negative_input_len, batch_trajs_len],
                [np.array(distance), np.array(negative_distance)])
            j = j + self.batch_size

    def trained_model_eval(self, print_batch=10, print_test=100, save_model=True, load_model=None,
                           in_cell_update=True, stard_LSTM=False):

        spatial_net = NeuTraj_Network(4, self.target_size, self.grid_size,
                                      self.batch_size, self.sampling_num,
                                      stard_LSTM=stard_LSTM, incell=in_cell_update)

        if load_model != None:
            m = torch.load(load_model)
            spatial_net.load_state_dict(m)

            embeddings = tm.test_comput_embeddings(
                self, spatial_net, test_batch=config.em_batch)
            # print('len(embeddings): {}'.format(len(embeddings)))
            # print(embeddings.shape)
            # print(embeddings[0].shape)

            acc1 = tm.test_model(self, spatial_net.lorentz, embeddings,
                                 test_range=range(len(self.train_seqs), len(self.train_seqs) + config.test_num),
                                 similarity=True, print_batch=print_test, r10in50=True)
            return acc1

    def neutraj_train(self, print_batch=10, print_test=100, save_model=True, load_model=None,
                      in_cell_update=True, stard_LSTM=False):
        date = datetime.now(timezone(timedelta(hours=8))).strftime("%H:%M-%m%d") + '.log'
        logging.basicConfig(level=logging.DEBUG,
                            format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                            handlers=[logging.FileHandler(
                                config.log_root + f"{config.fname}_{config.distance_type}_lorentz{config.lorentz}_{date}",
                                mode='w'), logging.StreamHandler()])
        # logging.info(config.config_to_str())



        spatial_net_ = NeuTraj_Network(4, self.target_size, self.grid_size, self.batch_size, self.sampling_num,
                                      stard_LSTM=stard_LSTM, incell=in_cell_update)
        #--------
        #########
        spatial_net = EmbeddingModelHandler(spatial_net_,
                                            lh_input_size=2,
                                            lh_target_size=self.target_size,
                                            lh_lorentz=config.lorentz,
                                            lh_trajs=self.coor_trajs,
                                            lh_model_type=config.lorentz_mod,
                                            lh_sqrt=config.sqrt,
                                            lh_C=config.C)
        trainer_lh = LH_trainer(spatial_net, config.lorentz, every_epoch=3, grad_reduce=0.1, loss_cmb=5)
        #########
        #--------

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, spatial_net.parameters()), lr=config.learning_rate)

        mse_loss_m = WeightedRankingLoss(batch_size=self.batch_size, sampling_num=self.sampling_num, lorentz=spatial_net.lorentz)
        if config.device == 'cuda':
            spatial_net.cuda()
            mse_loss_m.cuda()
        torch.autograd.set_detect_anomaly(True)
        # 您的模型训练代码...
        if load_model is not None:
            m = torch.load(open(load_model))
            spatial_net.load_state_dict(m)
            embeddings = tm.test_comput_embeddings(
                self, spatial_net, test_batch=config.em_batch)
            print('len(embeddings): {}'.format(len(embeddings)))
            print(embeddings.shape)
            print(embeddings[0].shape)

            tm.test_model(self, spatial_net.lorentz, embeddings,
                          test_range=range(len(self.train_seqs), len(self.train_seqs) + config.test_num),
                          similarity=True, print_batch=print_test, r10in50=True)
        best_50 = float("-inf")
        best_10 = float('-inf')
        best_5 = float('-inf')
        bestndcg = float('-inf')
        best10in50=float('-inf')
        early_stop = 0
        # print("start evaluation")



        for epoch in range(config.epochs):
            spatial_net.train()
            print("Start training Epochs : {}".format(epoch))

            #--------
            #########
            with trainer_lh.get_iter(epoch) as iter_lh:
                for train_str, loss_cmb in iter_lh:
            #########
            #--------
                    cnt = 0
                    sum_loss = 0
                    with tqdm(self.batch_generator(self.train_seqs, self.train_distance), train_str) as tq:
                        for i, batch in enumerate(tq):
                            inputs_arrays, inputs_len_arrays, target_arrays = batch[0], batch[1], batch[2]
                            idx = inputs_arrays[-1]

                            a_e, t_e, n_e = spatial_net(inputs_arrays, inputs_len_arrays)
                            positive_distance_target = torch.Tensor(target_arrays[0]).view((-1, 1))
                            negative_distance_target = torch.Tensor(target_arrays[1]).view((-1, 1))
                            loss = mse_loss_m(a_e, t_e, n_e, positive_distance_target, negative_distance_target, idx)
                            sum_loss += loss
                            cnt += 1
                            # assert loss.isnan() != True


                            if cnt % loss_cmb == loss_cmb - 1:
                                optimizer.zero_grad()
                                sum_loss.backward()

                                ##### add LH grad reduce
                                iter_lh.LH_grad_reduce()
                                #####

                                optimizer.step()
                                sum_loss=0

                            if not in_cell_update:
                                spatial_net.spatial_memory_update(inputs_arrays, inputs_len_arrays)

            with torch.no_grad():
                # spatial_net.lorentz.both_train(False, False)
                embeddings = tm.test_comput_embeddings(self, spatial_net, test_batch=config.em_batch)
                data_l = len(self.padded_trajs)

                #-------------
                ##############
                hr10, hr50, hr5, ndcg, _10in50, ret_metric, extra_msg = cal_top10_acc(self.distance, embeddings,
                                                      [int(data_l * config.seeds_radio), data_l],
                                                      spatial_net.lorentz, config.lorentz,_10in50=True, extra_msg=True)
                ##############
                #-------------

                print(f"gpu alloc:{torch.cuda.max_memory_allocated() / (1024 ** 3)}|"
                      f"resv:{torch.cuda.max_memory_reserved() / (1024 ** 3)}|cache:{torch.cuda.max_memory_cached() / (1024 ** 3)}")
                torch.save(spatial_net.state_dict(), f'./model/test_time_{config.lorentz}.h5')


            best_10 = max(hr10, best_10)
            best_5 = max(hr5, best_5)
            best_50 = max(hr50, best_50)
            bestndcg = max(ndcg, bestndcg)
            best10in50 = max(_10in50, best10in50)
            print(f"top5:{hr5}|{best_5},top10:{hr10}|{best_10},top50:{hr50}|{best_50},ndcg:{ndcg}|{bestndcg},top10in50:{_10in50}|{best10in50}")
            print(f"extra_msg:{extra_msg}")
            print(f"gpu alloc:{torch.cuda.max_memory_allocated() / (1024 ** 3)}|"
                f"resv:{torch.cuda.max_memory_reserved() / (1024 ** 3)}|cache:{torch.cuda.max_memory_cached() / (1024 ** 3)}")
            logging.info(
                f"epoch{epoch + 1}->top5:{hr5}|{best_5},top10:{hr10}|{best_10},top50:{hr50}|{best_50},ndcg:{ndcg}|{bestndcg},top10in50:{_10in50}|{best10in50}")
            # print(acc1)
            if best_10 == hr10:
                early_stop = 0
                save_model_name = f"./model/{config.data_type}_{config.distance_type}_{config.lorentz}_best_model.h5"
                print("better save:"+save_model_name)
                torch.save(spatial_net.state_dict(), save_model_name)
                pickle.dump(ret_metric, open(f"neutraj_metric_{config.fname}_{config.distance_type}_l{config.lorentz}_acc{hr10}", 'wb'))
            else:
                print("Worse!")
                early_stop += 1
                if early_stop >= 5:
                    print(f"Finish top5:{best_5},top10:{best_10} top50:{best_50}")
                    break;

import pickle

import numpy as np
from tqdm import tqdm
import time, os, sys
parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
print(parent_parent_dir)
sys.path.append(parent_parent_dir)
from lorenz.transfer import cal_top10_acc, cal_top10_acc_bak
chk=True
print('finish lorentz import')
from model_network import STTrajSimEncoder
print('finish model_network import')
import yaml
import torch
print('finish torch import')
import data_utils
from lossfun import LossFun
import test_method

print('finish loss import')

class STsim_Trainer(object):
    def __init__(self,cfg=None):
        config = yaml.safe_load(open('config.yaml'))
        if cfg is not None:
            config = yaml.safe_load(open(cfg))
        self.cfg=cfg
        self.feature_size = config["feature_size"]
        self.embedding_size = config["embedding_size"]
        self.date2vec_size = config["date2vec_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout_rate = config["dropout_rate"]
        self.concat = config["concat"]
        self.device = "cuda:" + str(config["cuda"])
        self.learning_rate = config["learning_rate"]
        self.epochs = config["epochs"]

        self.train_batch = config["train_batch"]
        self.test_batch = config["test_batch"]
        self.traj_file = str(config["traj_file"])
        self.time_file = str(config["time_file"])

        self.dataset = str(config["dataset"])
        self.distance_type = str(config["distance_type"])
        self.early_stop = config["early_stop"]

    def ST_eval(self, load_model=None):
        net = STTrajSimEncoder(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               date2vec_size=self.date2vec_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout_rate=self.dropout_rate,
                               concat=self.concat,
                               device=self.device)

        if load_model != None:
            net.load_state_dict(torch.load(load_model))
            net.to(self.device)

            dataload = data_utils.DataLoader()
            road_network = data_utils.load_netowrk(self.dataset).to(self.device)

            with torch.no_grad():
                vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='test')
                embedding_vali = test_method.compute_embedding(road_network=road_network, net=net,
                                                               test_traj=list(vali_node_list),
                                                               test_time=list(vali_d2vec_list),
                                                               test_batch=self.test_batch)
                acc = test_method.test_model(embedding_vali, isvali=False)
                print(acc)

    def ST_train(self, load_model=None, load_optimizer=None, lorenz=0, ratio=1.0):

        net = STTrajSimEncoder(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               date2vec_size=self.date2vec_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout_rate=self.dropout_rate,
                               concat=self.concat,
                               device=self.device,lorenz=lorenz)

        dataload = data_utils.DataLoader()
        dataload.get_triplets(cfg=self.cfg)
        node_list_int_, _, train_d2vec_list_ = dataload.load(load_part='train', cfg=self.cfg)
        vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='vali', cfg=self.cfg)
        node_list_int = np.concatenate((node_list_int_, vali_node_list))
        train_d2vec_list = np.concatenate((train_d2vec_list_, vali_d2vec_list))
        data_utils.triplet_groud_truth(self.cfg)

        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=self.learning_rate,
                                     weight_decay=0.0001)
        lossfunction = LossFun(self.train_batch, self.distance_type, net.lorenz,cfg=self.cfg)

        net.to(self.device)
        lossfunction.to(self.device)

        road_network = data_utils.load_netowrk(self.dataset).to(self.device)

        bt_num = int(dataload.return_triplets_num()*ratio / self.train_batch)

        batch_l = data_utils.batch_list(batch_size=self.train_batch, nodes=node_list_int, d2vecs=train_d2vec_list, cfg=self.cfg, ratio=ratio)

        config = yaml.safe_load(open(self.cfg))
        valid_dis_matrix = np.load(str(config["path_vali_truth"]))
        for i in range(4000):
            for j in range(4000):
                if valid_dis_matrix[i][j] == -1:
                    valid_dis_matrix[i][j] = np.inf

        best_epoch = 0
        best_hr10 = 0
        lastepoch = '0'
        if load_model != None:
            net.load_state_dict(torch.load(load_model))
            optimizer.load_state_dict(torch.load(load_optimizer))
            lastepoch = load_model.split('/')[-1].split('_')[3]
            best_epoch = int(lastepoch)

        for epoch in range(int(lastepoch), self.epochs):
            net.train()
            net.reload_traj(network=road_network, traj_seqs=node_list_int.tolist(), time_seqs=train_d2vec_list.tolist())
            if True:
                net.lorenz.both_train(True, False)
                with tqdm(range(bt_num), f'{epoch}:raw training') as t:
                    for bt in t:
                        a_node_batch, a_time_batch, p_node_batch, p_time_batch, n_node_batch, n_time_batch, batch_index, ids = batch_l.getbatch_one()
                        if chk:
                            for i in range(len(a_node_batch)):
                                l = min(len(node_list_int[ids[0][i]]), len(a_node_batch[i]))
                                assert node_list_int[ids[0][i]][:l] == a_node_batch[i][:l]
                                l = min(len(node_list_int[ids[1][i]]), len(p_node_batch[i]))
                                assert node_list_int[ids[1][i]][:l] == p_node_batch[i][:l]
                                l = min(len(node_list_int[ids[2][i]]), len(n_node_batch[i]))
                                assert node_list_int[ids[2][i]][:l] == n_node_batch[i][:l]
                        a_embedding = net(road_network, a_node_batch, a_time_batch)
                        p_embedding = net(road_network, p_node_batch, p_time_batch)
                        n_embedding = net(road_network, n_node_batch, n_time_batch)

                        loss = lossfunction(a_embedding, p_embedding, n_embedding, batch_index, ids)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                if net.lorenz.lorenz > 0:
                    net.lorenz.both_train(False, True)
                    if epoch % 4 == 3:
                        all_loss, loss_cnt = 0, 0
                        with tqdm(range(bt_num), 'lorenz training') as t:
                            for bt in t:
                                a_node_batch, a_time_batch, p_node_batch, p_time_batch, n_node_batch, n_time_batch, batch_index, ids = batch_l.getbatch_one()
                                if chk:
                                    for i in range(len(a_node_batch)):
                                        assert node_list_int[ids[0][i]] == a_node_batch[i]
                                        assert node_list_int[ids[1][i]] == p_node_batch[i]
                                        assert node_list_int[ids[2][i]] == n_node_batch[i]

                                a_embedding = net(road_network, a_node_batch, a_time_batch)
                                p_embedding = net(road_network, p_node_batch, p_time_batch)
                                n_embedding = net(road_network, n_node_batch, n_time_batch)
                                loss = lossfunction(a_embedding, p_embedding, n_embedding, batch_index, ids)


                                assert loss.isnan() != True
                                # optimizer.zero_grad()
                                # loss.backward()
                                # optimizer.step()
                                all_loss += loss
                                loss_cnt += 1
                                if loss_cnt % 5 == 0:
                                    optimizer.zero_grad()
                                    all_loss.backward()
                                    for param in net.lorenz.traj2vec.parameters():
                                        if param.grad is not None:
                                            param.grad.data *= 0.1
                                    optimizer.step()
                                    all_loss = 0
            if epoch % 2 == 0:
                # net.eval()
                net.lorenz.both_train(False, False)
                with torch.no_grad():
                    vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='vali',cfg=self.cfg)
                    t0=time.time()

                    embedding_vali = test_method.compute_embedding(road_network=road_network, net=net,
                                                                   test_traj=list(vali_node_list),
                                                                   test_time=list(vali_d2vec_list),
                                                                   test_batch=self.test_batch)





                    tmp_dis_matrix = np.zeros((14000, 14000), dtype=np.float64)
                    tmp_dis_matrix[10000:, 10000:] = valid_dis_matrix
                    tmp_embedding_vali = torch.zeros((14000, embedding_vali.shape[1]), dtype=embedding_vali.dtype,
                                                     device=embedding_vali.device)
                    tmp_embedding_vali[10000:] = embedding_vali
                    # if net.lorenz.lorenz == 0:
                    #     cal_top10_acc_bak(tmp_dis_matrix, tmp_embedding_vali, [10000, 14000],
                    #                      net.lorenz, net.lorenz.lorenz, _10in50=True)
                    #     # acc = test_method.test_model(embedding_vali, isvali=True)
                    #     # print('epoch:', epoch, acc[0], acc[1], acc[2], loss.item())

                    acc = cal_top10_acc(tmp_dis_matrix, tmp_embedding_vali, [10000, 14000],
                                        net.lorenz, net.lorenz.lorenz, _10in50=True)


                    print(f"gpu alloc:{torch.cuda.max_memory_allocated() / (1024 ** 3)}|"
                          f"resv:{torch.cuda.max_memory_reserved() / (1024 ** 3)}|cache:{torch.cuda.max_memory_cached() / (1024 ** 3)}")
                    torch.save(net.state_dict(), f'model_test_l{net.lorenz.lorenz}.pkl')
                    print(f'epoch:{epoch}, HR10:{acc[0]}, HR50:{acc[1]}, HR5:{acc[2]}, HR10in50:{acc[4]},ncdg:{acc[3]} Loss:{loss.item()}')
                    # save model
                    save_modelname = './model/{}_{}_2w_ST/{}_{}_epoch_{}_HR10_{}_HR50_{}_HR1050_{}_Loss_{}_L{}.pkl'.format(
                        self.dataset, self.distance_type,
                        self.dataset, self.distance_type, str(epoch), acc[0], acc[1], acc[2], loss.item(),net.lorenz.lorenz)
                    path_tmp = './model/{}_{}_2w_ST'.format(self.dataset, self.distance_type)
                    if os.path.exists(path_tmp) == False:
                        os.makedirs(path_tmp)
                    torch.save(net.state_dict(), save_modelname)
                    pickle.dump(acc[-1], open(f'metric_{self.dataset}_{self.distance_type}_l{net.lorenz.lorenz}_acc{acc[0]}', 'wb'))
                    if acc[0] > best_hr10:
                        best_hr10 = acc[0]
                        best_epoch = epoch
                    if epoch - best_epoch >= self.early_stop:
                        break
import pickle

import numpy as np
from tqdm import tqdm
import time, os, sys
parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_parent_dir)
from lorentz.handler import EmbeddingModelHandler, LH_trainer


# print(parent_parent_dir)

from lorentz.transfer import cal_top10_acc
chk=True
from model_network import STTrajSimEncoder
import yaml
import torch
import data_utils
from lossfun import LossFun
import test_method


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

    def ST_train(self, load_model=None, load_optimizer=None, lorentz=0, ratio=1.0):
        #print(f'{self.cfg} Lorentz:{lorentz} msg:{1}')
        net = STTrajSimEncoder(feature_size=self.feature_size,
                               embedding_size=self.embedding_size,
                               date2vec_size=self.date2vec_size,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               dropout_rate=self.dropout_rate,
                               concat=self.concat,
                               device=self.device,lorentz=lorentz)

        # self.lorentz = Lorentz(base_model=[self.stEncoder], dim=(128, 256), lorentz=lorentz, trajs=trajs, load=None,
        #                        model_type='lstm', sqrt=8)


        # ---------add
        ##########
        net = EmbeddingModelHandler(net,
                                    lh_input_size=128,
                                    lh_target_size=256,
                                    lh_lorentz=lorentz,
                                    lh_trajs=None)
        trainer_lh = LH_trainer(net, lorentz, every_epoch=4, grad_reduce=0.1)
        ##########
        # ---------


        dataload = data_utils.DataLoader()
        dataload.get_triplets(cfg=self.cfg)
        node_list_int_, _, train_d2vec_list_ = dataload.load(load_part='train', cfg=self.cfg)
        vali_node_list, vali_time_list, vali_d2vec_list = dataload.load(load_part='vali', cfg=self.cfg)
        node_list_int = np.concatenate((node_list_int_, vali_node_list))
        train_d2vec_list = np.concatenate((train_d2vec_list_, vali_d2vec_list))
        data_utils.triplet_groud_truth(self.cfg)

        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=self.learning_rate,
                                     weight_decay=0.0001)
        lossfunction = LossFun(self.train_batch, self.distance_type, net.lorentz,cfg=self.cfg)

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
                    valid_dis_matrix[i][j] = 1e10

        best_epoch = 0
        best_hr10 = 0
        lastepoch = '0'
        if load_model != None:
            net.load_state_dict(torch.load(load_model))
            optimizer.load_state_dict(torch.load(load_optimizer))
            lastepoch = load_model.split('/')[-1].split('_')[3]
            best_epoch = int(lastepoch)
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(int(lastepoch), self.epochs):
            net.train()


            # -------- add
            #########

            # There is a little difference with other model
            # cause the traj in ST2Vec is Embedding, every epoch need update
            net.traj_reload(network=road_network, traj_seqs=node_list_int.tolist(), time_seqs=train_d2vec_list.tolist())

            with trainer_lh.get_iter(epoch) as iter_lh:
                for train_str, loss_cmb in iter_lh:
            #########
            #--------
                    sum_loss, cnt = 0, 0
                    with tqdm(range(bt_num), f'{epoch}:{train_str} training') as t:
                        for bt in t:
                            a_node_batch, a_time_batch, p_node_batch, p_time_batch, n_node_batch, n_time_batch, batch_index, ids = batch_l.getbatch_one()
                            a_embedding = net(road_network, a_node_batch, a_time_batch)
                            p_embedding = net(road_network, p_node_batch, p_time_batch)
                            n_embedding = net(road_network, n_node_batch, n_time_batch)

                            loss = lossfunction(a_embedding, p_embedding, n_embedding, batch_index, ids)
                            sum_loss += loss
                            cnt += 1

                            if cnt % loss_cmb == loss_cmb-1:

                                optimizer.zero_grad()
                                sum_loss.backward()

                                ##### add LH grad reduce
                                iter_lh.LH_grad_reduce()
                                #####

                                optimizer.step()
                                sum_loss = 0


            if epoch % 2 == 0:
                # net.eval()
                net.lorentz.both_train(False, False)
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

                    #--------mod
                    #########
                    acc = cal_top10_acc(tmp_dis_matrix, tmp_embedding_vali, [10000, 14000],
                                        net.lorentz, net.lorentz.lorentz, _10in50=True,extra_msg=True)
                    #########
                    #--------
                    msg = acc[-1]
                    print(f'{self.cfg}  Lorentz:{lorentz}  msg:{msg}')
                    torch.save(net.state_dict(), f'model_test_l{net.lorentz.lorentz}.pkl')
                    print(f'epoch:{epoch}, HR10:{acc[0]}, HR50:{acc[1]}, HR5:{acc[2]}, HR10in50:{acc[4]},ncdg:{acc[3]} Loss:{loss.item()}')
                    # save model
                    save_modelname = './model/{}_{}_2w_ST/{}_{}_epoch_{}_HR10_{}_HR50_{}_HR1050_{}_Loss_{}_L{}.pkl'.format(
                        self.dataset, self.distance_type,
                        self.dataset, self.distance_type, str(epoch), acc[0], acc[1], acc[2], loss.item(),net.lorentz.lorentz)
                    path_tmp = './model/{}_{}_2w_ST'.format(self.dataset, self.distance_type)
                    if os.path.exists(path_tmp) == False:
                        os.makedirs(path_tmp)
                    torch.save(net.state_dict(), save_modelname)
                    pickle.dump(acc[-1], open(f'metric_{self.dataset}_{self.distance_type}_l{net.lorentz.lorentz}_acc{acc[0]}', 'wb'))
                    if acc[0] > best_hr10:
                        best_hr10 = acc[0]
                        best_epoch = epoch
                    if epoch - best_epoch >= self.early_stop:
                        break

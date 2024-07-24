import copy, os, sys
import logging
import pickle
import time
import datetime
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import dgl

from exp.exp_basic import ExpBasic
from model.network.graph_transformer import GraphTransformer

from utils.build_qtree import build_qtree
from utils.pre_embedding import get_pre_embedding

from utils.data_loader import TrajGraphDataLoader
from model.loss import WeightedRankingLoss
from model.accuracy_functions import get_embedding_acc

from utils.tools import pload, pdump

parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_parent_dir)

from lorenz.transfer import cal_top10_acc, Lorenz


def view_model_param(model):
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print("MODEL Total parameters:", total_param, "\n")
    return total_param


def get_statistic(trajs):
    # trajs = np.array(trajs)
    lons = []
    lats = []
    for traj in trajs:
        for node in traj:
            lon, lat = node
            lons.append(lon)
            lats.append(lat)
    return np.array(lons), np.array(lats)


def rewrite_cfg(trajs, config):
    lons, lats = get_statistic(trajs)
    config['x_range'] = [lons.min(), lons.max()]

    config['y_range'] = [lats.min(), lats.max()]
    config['data_features'] = [lons.mean(), lons.std(), lats.mean(), lats.std()]


class ExpGraphTransformer(ExpBasic):
    def __init__(self, config, gpu_id, load_model, just_embeddings):
        self.load_model = load_model
        self.store_embeddings = just_embeddings
        self.lorenz = None

        super(ExpGraphTransformer, self).__init__(config, gpu_id)

        if just_embeddings:  # 只进行embedding操作
            trajs = pload(self.config["traj_path"].format(self.config["data"]))
            rewrite_cfg(trajs, self.config)
            print(f"refresh {self.config}")
            self.qtree = build_qtree(trajs, self.config["x_range"], self.config["y_range"], self.config["max_nodes"],
                                     self.config["max_depth"])
            # 决定是否要进行 embedding预训练
            self.qtree_name2id, self.pre_embedding = get_pre_embedding(self.qtree, self.config["d_model"])
            self.embeding_loader = self._get_dataloader(flag="embed")
            print("Embedding Graphs: ", len(self.embeding_loader.dataset))
        else:
            # self.log_writer = SummaryWriter(
            #     f"./runs/{self.config['data']}_{self.config['lorenz']}_{self.config['model']}_{self.config['dis_type']}_{datetime.datetime.now()}/")
            trajs = pload(self.config["traj_path"].format(self.config["data"]))
            rewrite_cfg(trajs, self.config)
            print("[!] Build qtree, max nodes:", self.config["max_nodes"], "max depth:", self.config["max_depth"],
                  "x_range:", self.config["x_range"], "y_range:", self.config["y_range"])

            point2vector_path = f"./{self.config['data']}_point2vector"

            # if os.path.exists(point2vector_path):
            if False:
                self.qtree, self.qtree_name2id, self.pre_embedding = pload(point2vector_path)
            else:
                self.qtree = build_qtree(trajs, self.config["x_range"], self.config["y_range"],
                                         self.config["max_nodes"], self.config["max_depth"])
                self.qtree_name2id, self.pre_embedding = get_pre_embedding(self.qtree, self.config["d_model"])
                pdump((self.qtree, self.qtree_name2id, self.pre_embedding), point2vector_path)
            self.train_loader = self._get_dataloader(flag="train")
            print("Training Graphs: ", len(self.train_loader.dataset))

            self.val_loader = self._get_dataloader(flag="val")
            print("Validation Graphs: ", len(self.val_loader.dataset))

        self.model = self._build_model(trajs).to(self.device)

    def _build_model(self,trajs):
        if self.config["model"] == "TrajGAT":
            model = GraphTransformer(d_input=self.config["d_input"], d_model=self.config["d_model"],
                                     num_head=self.config["num_head"],
                                     num_encoder_layers=self.config["num_encoder_layers"],
                                     d_lap_pos=self.config["d_lap_pos"], encoder_dropout=self.config["encoder_dropout"],
                                     layer_norm=self.config["layer_norm"], batch_norm=self.config["batch_norm"],
                                     in_feat_dropout=self.config["in_feat_dropout"], lorenz=self.config['lorenz'],
                                     pre_embedding=self.pre_embedding, trajs=trajs, config=self.config)  # 预训练得到的，每个结点的 structure embedding

        view_model_param(model)

        # self.lorenz = Lorenz(base_model=[model], dim=(2, self.config["d_model"]), lorenz=self.config['lorenz'], trajs=trajs,
        #                      load=None, model_type=self.config["model_type"])
        if self.load_model is not None:
            model.load_state_dict(torch.load(self.load_model))
            print("[!] Load model weight:", self.load_model)

        return model

    def _get_dataloader(self, flag):
        if flag == "train":
            trajs = pload(self.config["traj_path"].format(self.config["data"]))[
                    self.config["train_data_range"][0]: self.config["train_data_range"][1]]
            print("Train traj number:", len(trajs))
            print(f'traj_path:{self.config["traj_path"].format(self.config["data"])},matrix path{self.config["dis_matrix_path"].format(self.config["data"], self.config["dis_type"])}')
            matrix = pload(self.config["dis_matrix_path"].format(self.config["data"], self.config["dis_type"]))
            matrix = torch.tensor(matrix)[
                     self.config["train_data_range"][0]: self.config["train_data_range"][1],
                     self.config["train_data_range"][0]: self.config["train_data_range"][1]]
            # print("Train matrix shape:", matrix.shape)
            # print(matrix[:5, :5])
        elif flag == "val":
            trajs = pload(self.config["traj_path"].format(self.config["data"]))[
                    self.config["val_data_range"][0]: self.config["val_data_range"][1]]
            print("Val traj number:", len(trajs))
            # matrix = pload(self.config["dis_matrix_path"].format(self.config["data"], self.config["dis_type"]))[
            #          self.config["val_data_range"][0]: self.config["val_data_range"][1], :]
            matrix = pload(self.config["dis_matrix_path"].format(self.config["data"], self.config["dis_type"]))
            matrix = torch.tensor(matrix)
            # print("Val matrix shape:", matrix.shape)
            # print(matrix[:5, :5])
            # print(matrix[:, 6000:10000])

        elif flag == "embed":
            trajs = pload(self.config["traj_path"].format(self.config["data"]))[
                    self.config["emb_data_range"][0]: self.config["emb_data_range"][1]]
            matrix = torch.tensor(
                pload(self.config["dis_matrix_path"].format(self.config["data"], self.config["dis_type"])))[
                     self.config["emb_data_range"][0]: self.config["emb_data_range"][1],
                     self.config["emb_data_range"][0]: self.config["emb_data_range"][1]]

        data_loader = TrajGraphDataLoader(traj_data=trajs, dis_matrix=matrix, phase=flag,
                                          train_batch_size=self.config["train_batch_size"],
                                          eval_batch_size=self.config["eval_batch_size"],
                                          d_lap_pos=self.config["d_lap_pos"], sample_num=self.config["sample_num"],
                                          num_workers=self.config["num_workers"],
                                          data_features=self.config["data_features"], x_range=self.config["x_range"],
                                          y_range=self.config["y_range"], qtree=self.qtree,
                                          qtree_name2id=self.qtree_name2id).get_data_loader()

        return data_loader

    def _select_optimizer(self):
        if self.config["optimizer"] == "SGD":
            model_optim = optim.SGD(self.model.parameters(), lr=self.config["init_lr"])
        elif self.config["optimizer"] == "Adam":
            model_optim = optim.Adam(self.model.parameters(), lr=self.config["init_lr"])

        return model_optim, None

    def _select_criterion(self):
        criterion = WeightedRankingLoss(self.config["sample_num"], self.config["alpha"], self.device, self.model.lorenz)
        return criterion

    def embedding(self):
        all_vectors = []
        self.model.eval()

        loader_time = 0
        begin_time = time.time()
        mark_time = time.time()
        for trajgraph_l_l, _ in tqdm(self.embeding_loader):
            loader_time += time.time() - mark_time
            # trajgraph_l_l [B, 1, graph]
            B = len(trajgraph_l_l)
            D = self.config["d_model"]

            traj_graph = []
            for b in trajgraph_l_l:
                traj_graph.extend(b)
            batch_graphs = dgl.batch(traj_graph).to(self.device)  # (B, graph)

            with torch.no_grad():
                vectors = self.model(batch_graphs)  # vecters [B, d_model]

            all_vectors.append(vectors)
            mark_time = time.time()

        all_vectors = torch.cat(all_vectors, dim=0)
        print("all_embeding_vectors length:", len(all_vectors))
        print("all_embedding_vectors shape:", all_vectors.shape)

        end_time = time.time()
        print(f"all embedding time: {end_time - begin_time - loader_time} seconds")

        # pdump(all_vectors, f"{self.config['length']}_{self.config['dis_type']}_embeddings_{all_vectors.shape[0]}_{all_vectors.shape[1]}.pkl")

        # hr10, hr50, r10_50 = get_embedding_acc(row_embedding_tensor=all_vectors, col_embedding_tensor=all_vectors,
        #                                        distance_matrix=self.embeding_loader.dataset.dis_matrix,
        #                                        matrix_cal_batch=self.config["matrix_cal_batch"], )
        hr10, hr50, r10_50,_ = cal_top10_acc(self.val_loader.dataset.dis_matrix, all_vectors,
                                           self.config["val_data_range"], self.model.lorenz, self.config['lorenz'])

        print(hr10, hr50, r10_50)

    def val(self):
        all_vectors = []
        all_id = []
        self.model.eval()
        for trajgraph_l_l, _, idx_l_l in self.val_loader:
            # trajgraph_l_l [B, 1, graph]
            B = len(trajgraph_l_l)
            D = self.config["d_model"]

            traj_graph = []
            idx_l = []
            for i,b in enumerate(trajgraph_l_l):
                traj_graph.extend(b)
                idx_l.extend(idx_l_l[i])
            batch_graphs = dgl.batch(traj_graph).to(self.device)  # (B, graph)

            with torch.no_grad():
                vectors = self.model(batch_graphs)  # vecters [B, d_model]


            all_vectors.append(vectors)
            all_id.extend(idx_l)
        all_vectors = torch.cat(all_vectors, dim=0)
        # print("all_val_vectors length:", len(all_vectors))

        # hr10, hr50, r10_50 = get_embedding_acc(
        #     row_embedding_tensor=all_vectors[self.config["val_data_range"][0]: self.config["val_data_range"][1]],
        #     col_embedding_tensor=all_vectors, distance_matrix=self.val_loader.dataset.dis_matrix,
        #     matrix_cal_batch=self.config["matrix_cal_batch"], )
        new_vectors = torch.zeros((10000, all_vectors.shape[-1]), dtype=torch.float32, device='cpu')
        new_vectors[self.config["val_data_range"][0]:self.config["val_data_range"][1]] = all_vectors

        hr10, hr50, hr5, ndcg, metric = cal_top10_acc(self.val_loader.dataset.dis_matrix.numpy(), new_vectors, self.config["val_data_range"],
                                              self.model.lorenz, self.config['lorenz'],quick=False)
        self.model.train()
        return hr10, hr50, hr5, ndcg, metric

    def train(self):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_hr10, best_hr5, best_hr50, best_ndcg = 0.0, 0.0, 0.0, 0.0
        time_now = time.time()
        early_stop = 0
        model_optim, scheduler = self._select_optimizer()
        criterion = self._select_criterion()
        # best_hr10, best_hr50, best_hr5, best_ndcg = self.val()
        for epoch in range(self.config["epoch"]):
            if True:
                self.model.train()

                epoch_begin_time = time.time()
                epoch_loss = 0.0

                dataload_time = 0
                embed_time = 0
                groupdata_time = 0
                test_time = time.time()

                self.model.lorenz.both_train(True, False)
                with tqdm(self.train_loader,'raw') as tq:
                    for trajgraph_l_l, dis_l, idx_l in tq:
                        dataload_time += time.time() - test_time
                        test_time2 = time.time()
                        # trajgraph_l_l [B, SAM, graph]
                        # dis_l [B, SAM]
                        B = len(trajgraph_l_l)
                        SAM = self.config["sample_num"]
                        D = self.config["d_model"]

                        traj_graph = []
                        for b in trajgraph_l_l:
                            traj_graph.extend(b)
                        batch_graphs = dgl.batch(traj_graph).to(self.device)  # (B*SAM, graph)
                        groupdata_time += time.time() - test_time2
                        test_time3 = time.time()
                        model_optim.zero_grad()

                        with torch.set_grad_enabled(True):
                            vectors = self.model(batch_graphs)  # vecters [B*SAM, d_model]

                        vectors = vectors.view(B, SAM, D)

                        loss = criterion(vectors, torch.tensor(dis_l).to(self.device), torch.tensor(idx_l))

                        loss.backward()
                        model_optim.step()

                        epoch_loss += loss.item()
                        embed_time += time.time() - test_time3
                        test_time = time.time()

                print("\nLoad data time:", dataload_time // 60, "m")
                print("Data group time:", groupdata_time // 60, "m")
                print("Train model time:", embed_time // 60, "m\n")

                epoch_loss = epoch_loss / len(self.train_loader.dataset)
                # self.log_writer.add_scalar(f"TrajRepresentation/Loss", float(epoch_loss), epoch)

                if self.model.lorenz.lorenz != 0:
                    if epoch %3 == 2:
                        self.model.lorenz.both_train(False, True)
                        with tqdm(self.train_loader,'lorentz') as tq:
                            for trajgraph_l_l, dis_l, idx_l in tq:
                                B = len(trajgraph_l_l)
                                SAM = self.config["sample_num"]
                                D = self.config["d_model"]

                                traj_graph = []
                                for b in trajgraph_l_l:
                                    traj_graph.extend(b)
                                batch_graphs = dgl.batch(traj_graph).to(self.device)  # (B*SAM, graph)
                                model_optim.zero_grad()
                                with torch.set_grad_enabled(True):
                                    vectors = self.model(batch_graphs)  # vecters [B*SAM, d_model]
                                    vectors = vectors.view(B, SAM, D)
                                    loss = criterion(vectors, torch.tensor(dis_l).to(self.device), torch.tensor(idx_l))

                                    for param in self.model.lorenz.traj2vec.parameters():
                                        if param.grad is not None:
                                            param.grad.data *= 0.1

                                    loss.backward()
                                    model_optim.step()
                # scheduler.step(epoch_loss)
                epoch_end_time = time.time()
                print(
                    f"\nEpoch {epoch + 1}/{self.config['epoch']}:\nTrain Loss: {epoch_loss:.4f}\tTime: {(epoch_end_time - epoch_begin_time) // 60} m {int((epoch_end_time - epoch_begin_time) % 60)} s")

                #val_begin_time = time.time()
            print('start val!!!!!!!!!!!!!!!!!!!!!!!!=============================')
            hr10, hr50, hr5, ndcg, metric = self.val()
            print(f"gpu alloc:{torch.cuda.max_memory_allocated() / (1024 ** 3)}|"
                  f"resv:{torch.cuda.max_memory_reserved() / (1024 ** 3)}|cache:{torch.cuda.max_memory_cached() / (1024 ** 3)}")
            better_str = "better" if hr10 >= best_hr10 else "worse_"
            best_hr5 = max(hr5, best_hr5)
            best_hr10 = max(hr10, best_hr10)
            best_hr50 = max(hr50, best_hr50)
            best_ndcg = max(ndcg, best_ndcg)
            print(
                f"{better_str} Val HR10: {100 * hr10:.4f}|{100 * best_hr10:.4f}\tHR50: {100 * hr50:.4f}|{100 * best_hr50:.4f}%\tHR5: {100 * hr5:.4f}|{100 * best_hr5:.4f}%\tndcg:{ndcg:.7f}|{best_ndcg:.7f}")
            logging.info(
                f"{better_str} Val HR10: {100 * hr10:.4f}|{100 * best_hr10:.4f}\tHR50: {100 * hr50:.4f}|{100 * best_hr50:.4f}%\tHR5: {100 * hr5:.4f}|{100 * best_hr5:.4f}%\tndcg:{ndcg:.7f}|{best_ndcg:.7f}")
            if hr10 >= best_hr10:
                early_stop = 0
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(best_model_wts,
                           self.config["model_best_wts_path"].format(self.config["data"], self.config["lorenz"],
                                                                     self.config["model"],
                                                                     self.config["dis_type"], best_hr10))
                # pickle.dump(metric, open(f"gat_metric_{self.config['data']}_{self.config['dis_type']}_l{self.config['lorenz']}_acc{best_hr10}", "wb"))
            else:
                early_stop += 1
                if early_stop >= 5:
                    break
        time_end = time.time()

        print("\nAll training complete in {:.0f}m {:.0f}s".format((time_end - time_now) // 60,
                                                                  (time_end - time_now) % 60))

        logging.info(
            f"Best Val HR10: {100 * best_hr10:.4f}%\tHR50: {100 * best_hr50:.4f}%\tHR5: {100 * best_hr5:.4f}%\tndcg: {best_ndcg}")

        torch.save(best_model_wts, self.config["model_best_wts_path"].format(self.config["data"], self.config["lorenz"],
                                                                             self.config["model"],
                                                                             self.config["dis_type"], best_hr10))

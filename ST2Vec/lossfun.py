from torch.nn import Module, Parameter, PairwiseDistance
import torch
import numpy as np
import math
import yaml
import pandas as pd

class LossFun(Module):
    def __init__(self,train_batch,distance_type,lorentz=None,cfg=None):
        super(LossFun, self).__init__()
        self.train_batch = train_batch
        self.distance_type = distance_type
        if cfg is not None:
            config = yaml.safe_load(open(cfg))
        else:
            config = yaml.safe_load(open('config.yaml'))
        self.triplets_dis = np.load(str(config["path_triplets_truth"]))
        self.lorentz = lorentz

    def forward(self,embedding_a,embedding_p,embedding_n,batch_index, ids):
        a_ids, p_ids, n_ids = ids
        batch_triplet_dis = self.triplets_dis[batch_index]
        batch_loss = 0.0

        for i in range(self.train_batch):

            D_ap = math.exp(-batch_triplet_dis[i][0])
            D_an = math.exp(-batch_triplet_dis[i][1])

            v_ap_ = torch.exp(-torch.dist(embedding_a[i], embedding_p[i], p=2))
            v_an_ = torch.exp(-torch.dist(embedding_a[i], embedding_n[i], p=2))

            if self.lorentz.lorenz == 0:
                pairdist = PairwiseDistance(p=2)
                v_ap = pairdist(embedding_a[i], embedding_p[i])
                v_an = pairdist(embedding_a[i], embedding_n[i])
                v_ap = torch.exp(-v_ap)
                v_an = torch.exp(-v_an)

            else:
                # dis_pred = self.lorenz.dist(anchor_trajs, traj_list)
                v_ap = self.lorentz.learned_cmb_dist(embedding_a[i], embedding_p[i], traj_i=a_ids[i], traj_j=p_ids[i])
                v_ap = torch.exp(-v_ap)
                v_an = self.lorentz.learned_cmb_dist(embedding_a[i], embedding_n[i], traj_i=a_ids[i], traj_j=n_ids[i])
                v_an = torch.exp(-v_an)

            loss_entire_ap = D_ap * ((D_ap - v_ap) ** 2)
            loss_entire_an = D_an * ((D_an - v_an) ** 2)

            oneloss = loss_entire_ap + loss_entire_an
            batch_loss += oneloss

        mean_batch_loss = batch_loss / self.train_batch
        sum_batch_loss = batch_loss

        return mean_batch_loss


import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn import Module, Parameter
import torch
import tools.config as config
import numpy as np


class WeightMSELoss(Module):
    def __init__(self, batch_size, sampling_num):
        super(WeightMSELoss, self).__init__()
        self.weight = []
        for i in range(batch_size):
            self.weight.append(0.)
            for traj_index in range(sampling_num):
                self.weight.append(np.array([config.sampling_num - traj_index]))

        self.weight = np.array(self.weight, dtype=object).astype(np.float32)
        sum = np.sum(self.weight)
        self.weight = self.weight / sum
        if config.device == 'cuda':
            self.weight = Parameter(torch.Tensor(self.weight).cuda(), requires_grad=False)
        else:
            self.weight = Parameter(torch.Tensor(self.weight).cpu(), requires_grad=False)
        self.batch_size = batch_size
        self.sampling_num = sampling_num

    def forward(self, input, target, isReLU=False):
        div = target - input.view(-1, 1)
        if isReLU:
            div = F.relu(div.view(-1, 1))
        square = torch.mul(div.view(-1, 1), div.view(-1, 1))
        weight_square = torch.mul(square.view(-1, 1), self.weight.view(-1, 1))

        loss = torch.sum(weight_square)
        return loss


class WeightedRankingLoss(Module):
    def __init__(self, batch_size, sampling_num, lorentz=None):
        super(WeightedRankingLoss, self).__init__()
        self.positive_loss = WeightMSELoss(batch_size, sampling_num)
        self.negative_loss = WeightMSELoss(batch_size, sampling_num)
        self.lorentz = lorentz

    def forward(self,anchor_embedding, trajs_embedding, negative_embedding, p_target, n_target, idx):
        anchors_idx, trajs_idx, negs_idx= idx
        #-------------- using learned_cmb_dist
        ###############
        if self.lorentz == 0:
            positive = torch.exp(-F.pairwise_distance(anchor_embedding, trajs_embedding, p=2))
            negative = torch.exp(-F.pairwise_distance(anchor_embedding, negative_embedding, p=2))
        else:
            positive = torch.exp(-self.lorentz.learned_cmb_dist(anchor_embedding, trajs_embedding, anchors_idx, trajs_idx))
            negative = torch.exp(-self.lorentz.learned_cmb_dist(anchor_embedding, negative_embedding, anchors_idx, negs_idx))
        ###############
        #--------------
        p_input = positive
        n_input = negative
        if config.device == 'cuda':
            trajs_mse_loss = self.positive_loss(p_input, autograd.Variable(p_target).cuda(), False)
            negative_mse_loss = self.negative_loss(n_input, autograd.Variable(n_target).cuda(), True)
        else:
            trajs_mse_loss = self.positive_loss(p_input, autograd.Variable(p_target).cpu(), False)
            negative_mse_loss = self.negative_loss(n_input, autograd.Variable(n_target).cpu(), True)
        self.trajs_mse_loss = trajs_mse_loss
        self.negative_mse_loss = negative_mse_loss
        loss = sum([trajs_mse_loss, negative_mse_loss])
        return loss

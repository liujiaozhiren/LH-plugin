import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lorenz.lorenz_net import Traj2Vector
# from neutraj.tools import config
import time

n = 4


class Lorenz(nn.Module):
    def __init__(self, dim, base_model, C=1, alpha=0.1, lorenz=0, trajs=None, load=None, model_type='lstm', sqrt=8.0,
                 net_init=None):
        super(Lorenz, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 将 C 和 alpha 转换为可学习的参数
        self.C = C
        dim_in, dim = dim
        self.dim = dim
        # self.alpha = nn.Parameter(torch.tensor([alpha], dtype=torch.float, device=self.device))
        self.lorenz = lorenz
        if lorenz != 0:
            self.linear = nn.Linear(dim, dim, device=self.device, dtype=torch.float64)
            self.linear_ex = nn.Linear(dim, dim, device=self.device, dtype=torch.float64)

            self.traj2vec = Traj2Vector(dim_in, dim // 4, trajs=trajs, device=self.device, model_type=model_type,
                                        load=load, init=net_init, )
        else:
            self.traj2vec = nn.Identity()
            self.linear = nn.Identity()
            self.linear_ex = nn.Identity()
        self.tmp_data = None
        self.base_model = base_model
        self.base_model.append(self.linear)
        self.base_model.append(self.linear_ex)
        self.sqrt = sqrt
        # self.linear = nn.LayerNorm(dim, device=self.device, dtype=torch.float64)

    def dynamic_store_traj_data(self, trajs, lens):
        if self.lorenz == 0:
            return
        self.traj2vec.dynamic_store_traj_data(trajs, lens=lens)

    def both_train(self, base=True, ration=True):
        for model in self.base_model:
            model.train(base)
            for param in model.parameters():
                param.requires_grad = base
        self.traj2vec.train(ration)
        for param in self.traj2vec.parameters():
            param.requires_grad = ration

    def __cosh(self, input1, input2):
        # concate input1 and input2 in dim 0
        # print(input1.shape, input2.shape)
        shape = input1.shape
        tmp = torch.cat([input1, input2], dim=0)
        ret = self._cosh_encode(tmp)
        # test = self._cosh_encode(input1)
        # print(input1.shape, input2.shape, ret.shape, test.shape, ret[:shape[0]//2].shape)
        # exit()
        return ret[:shape[0]], ret[shape[0]:]

    def _cosh_encode(self, input) -> torch.Tensor:
        # assert self.C == 1
        sq_c = math.sqrt(self.C)
        target_shape = (*input.shape[:-1], input.shape[-1] + 1)
        new_input = torch.zeros(target_shape, dtype=input.dtype, device=input.device)
        new_input[..., 0] = sq_c

        dist = torch.sum(input * input, dim=-1).pow(1 / self.sqrt)

        target_dist = torch.sinh(dist)
        dist_, target_dist_ = dist.unsqueeze(-1), target_dist.unsqueeze(-1)
        new_input[..., 1:] = (target_dist_ / dist_) * input
        new_input[..., 0] = torch.norm(new_input, dim=-1, p=2)
        # output[..., 1:] = new_input

        return new_input


        # assert self.C == 1
        target_shape = (*input.shape[:-1], input.shape[-1] + 1)
        output = torch.zeros(target_shape, dtype=input.dtype, device=input.device)
        dist = torch.sum(input * input, dim=-1).pow(1 / self.sqrt)
        assert (input.isnan()).any() == False
        assert (dist >= 0).all()
        # z = torch.cosh(dist)
        target_dist = torch.sinh(dist)
        dist_, target_dist_ = dist.unsqueeze(-1), target_dist.unsqueeze(-1)
        tmp_time = time.time()
        new_input = input / dist_ * target_dist_
        output[..., 0] = torch.sqrt(torch.sum(new_input ** 2, dim=-1) + self.C)
        output[..., 1:] = new_input
        # assert ((torch.sum(output[..., 1:] * output[..., 1:], dim=-1) - output[..., 0] * output[
        #     ..., 0] + self.C) < 1e-5).all()

    # def _cosh_encode2(self, input) -> torch.Tensor:
    #     assert self.C == 1
    #     target_shape = (*input.shape[:-1], input.shape[-1] + 1)
    #     output = torch.zeros(target_shape, dtype=input.dtype, device=input.device)
    #     div = self.lorenz + 1
    #     if div == 0:
    #         dist = torch.log(torch.sum(input * input, dim=-1) + 1)
    #     else:
    #         dist = torch.sum(input * input, dim=-1).pow(1.0 / div)
    #     assert (input.isnan()).any() == False
    #     assert (dist >= 0).all()
    #     # z = torch.cosh(dist)
    #     target_dist = torch.sinh(dist)
    #     new_input = input / dist.unsqueeze(-1) * target_dist.unsqueeze(-1)
    #     output[..., 0] = torch.sqrt(torch.sum(new_input ** 2, dim=-1) + self.C)
    #     output[..., 1:] = new_input
    #     # assert ((torch.sum(output[..., 1:] * output[..., 1:], dim=-1) - output[..., 0] * output[
    #     #     ..., 0] + self.C) < 1e-5).all()
    #     return output

    def _simple_encode(self, input) -> torch.Tensor:
        # ori_type = input.device
        if type(input) != torch.Tensor:
            input = torch.tensor(input)
        target_shape = (*input.shape[:-1], input.shape[-1] + 1)
        output = torch.zeros(target_shape, dtype=input.dtype, device=input.device)
        output[..., 1:] = input
        output[..., 0] = torch.sqrt(torch.sum(input ** 2, dim=-1) + self.C)

        return output

    def _lorenz_distance(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        assert batch_x.shape == batch_y.shape
        mm = batch_x * batch_y
        mm[..., 0] = - mm[..., 0]
        rst = -torch.sum(mm, dim=-1)
        return rst

    def _simple_lorenz_dist(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        lorenz_x = self._simple_encode(batch_x)
        lorenz_y = self._simple_encode(batch_y)
        # lorenz_dist = torch.abs(self.alpha.to(batch_x.device)) * (self._lorenz_distance(lorenz_x, lorenz_y) - self.C)
        lorenz_dist = self._lorenz_distance(lorenz_x, lorenz_y) - self.C
        return lorenz_dist

    def _cosh_lorenz_dist(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:

        # lorenz_x = self._cosh_encode(batch_x)
        # lorenz_y = self._cosh_encode(batch_y)
        lorenz_x, lorenz_y = self.__cosh(batch_x, batch_y)

        # lorenz_dist = torch.abs(self.alpha.to(batch_x.device)) * (self._lorenz_distance(lorenz_x, lorenz_y) - self.C)
        lorenz_dist = self._lorenz_distance(lorenz_x, lorenz_y) - self.C
        assert torch.exp(-lorenz_dist).isinf().any() == False
        return lorenz_dist

    def _cosh_lorenz_dist2(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        lorenz_x = self._cosh_encode2(batch_x)
        lorenz_y = self._cosh_encode2(batch_y)
        # lorenz_dist = torch.abs(self.alpha.to(batch_x.device)) * (self._lorenz_distance(lorenz_x, lorenz_y) - self.C)
        lorenz_dist = self._lorenz_distance(lorenz_x, lorenz_y) - self.C
        assert torch.exp(-lorenz_dist).isinf().any() == False
        return lorenz_dist

    def _normal_dist(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> torch.Tensor:
        if type(batch_x) != torch.Tensor:
            batch_x, batch_y = torch.tensor(batch_x), torch.tensor(batch_y)
        return ((batch_x - batch_y) ** 2).sum(-1)
        # return F.pairwise_distance(batch_x, batch_y, p=2)

    def dist(self, batch_x: torch.Tensor, batch_y: torch.Tensor, config_lorenz=None) -> torch.Tensor:
        if self.lorenz == 0:
            _batch_x = batch_x.to(torch.float64)
            _batch_y = batch_y.to(torch.float64)
            return self._normal_dist(_batch_x, _batch_y)
        return self.cmb_dist(batch_x, batch_y, config_lorenz)

    # def ori_dist(self, batch_x: torch.Tensor, batch_y: torch.Tensor, config_lorenz=None) -> torch.Tensor:
    #     _batch_x = batch_x.to(torch.float64)
    #     _batch_y = batch_y.to(torch.float64)
    #
    #     ori_device = batch_x.device
    #     batch_x = self.linear(_batch_x.to(self.device)).to(ori_device)
    #     batch_y = self.linear(_batch_y.to(self.device)).to(ori_device)
    #     if config_lorenz is not None:
    #         assert self.lorenz == config_lorenz
    #     if self.lorenz == 1:
    #         return self._simple_lorenz_dist(batch_x, batch_y)
    #     # elif self.lorenz == 2:
    #     #     return self._cosh_lorenz_dist(batch_x, batch_y)
    #     elif self.lorenz == 0:
    #         return self._normal_dist(_batch_x, _batch_y)
    #     elif self.lorenz >= 2:
    #         return self._cosh_lorenz_dist2(batch_x, batch_y)
    #     elif self.lorenz == -1:
    #         return self._cosh_lorenz_dist2(batch_x, batch_y)
    #     else:
    #         raise Exception("not support")

    def cmb_dist(self, batch_x: torch.Tensor, batch_y: torch.Tensor, config_lorenz=None) -> torch.Tensor:
        _batch_x = batch_x.to(torch.float64)
        _batch_y = batch_y.to(torch.float64)

        ori_device = batch_x.device
        batch_x1 = self.linear(_batch_x.to(self.device)).to(ori_device)
        batch_x2 = self.linear_ex(_batch_x.to(self.device)).to(ori_device)
        batch_y1 = self.linear(_batch_y.to(self.device)).to(ori_device)
        batch_y2 = self.linear_ex(_batch_y.to(self.device)).to(ori_device)

        num_1or2 = int(self.lorenz * self.dim / 2)
        num_1or2 = 0 if num_1or2 <= 1 else num_1or2
        num_0 = self.dim - num_1or2 * 2
        batch_x0, batch_y0 = _batch_x[..., :num_0], _batch_y[..., :num_0]
        batch_x1, batch_y1 = batch_x1[..., :num_1or2], batch_y1[..., :num_1or2]
        batch_x2, batch_y2 = batch_x2[..., :num_1or2], batch_y2[..., :num_1or2]
        dist0 = self._normal_dist(batch_x0, batch_y0)
        dist1 = self._simple_lorenz_dist(batch_x1, batch_y1)
        dist2 = self._cosh_lorenz_dist(batch_x2, batch_y2)
        if num_1or2 == 0:
            return dist0
        if num_0 == 0:
            return dist1 + dist2
        return dist0 + dist1 + dist2

    def gen_valid_ratio_emb(self, _range):
        if self.lorenz != 0:
            self.traj2vec.gen_valid_ratio_emb(_range)

    def learned_cmb_dist(self, batch_x: torch.Tensor, batch_y: torch.Tensor, traj_i=None, traj_j=None,
                         train=True, default=False) -> torch.Tensor:
        if self.lorenz == 0:
            return self._normal_dist(batch_x, batch_y)

        _batch_x = batch_x.to(torch.float64)
        _batch_y = batch_y.to(torch.float64)

        ori_device = batch_x.device
        # batch_x1 = self.linear(_batch_x.to(self.device)).to(ori_device)
        batch_x2 = self.linear_ex(_batch_x.to(self.device))
        # batch_y1 = self.linear(_batch_y.to(self.device)).to(ori_device)
        batch_y2 = self.linear_ex(_batch_y.to(self.device))
        if self.lorenz < 0 or default:
            ratio0, ratio1, ratio2 = torch.tensor(0.25), torch.tensor(0.75 / 2), torch.tensor(0.75 / 2)
        else:
            if train:
                ratio0, ratio1, ratio2 = self.traj2vec.gen_ration(traj_i, traj_j)
            else:
                ratio0, ratio1, ratio2 = self.traj2vec.get_ration(traj_i, traj_j)
        ratio1 = 0 if ratio1 is None else ratio1
        assert ((ratio0 + ratio1 + ratio2 - 1) ** 2 < 1e-8).all()
        dist0 = self._normal_dist(_batch_x, _batch_y).cpu()
        # dist1 = self._simple_lorenz_dist(batch_x1, batch_y1)
        dist2 = self._cosh_lorenz_dist(batch_x2, batch_y2).cpu()
        dev = dist0.device
        # return dist0 * ratio0.to(dev) + dist1 * ratio1.to(dev) + dist2 * ratio2.to(dev)
        return dist0 * ratio0.to(dev) + dist2 * ratio2.to(dev)

    # def store_traj_data(self,trajs):
    #     self.traj_data = torch.Tensor(trajs).to(self.device)
    #     trajs_len = []
    #     for traj in trajs:
    #         traj_len = len(traj)
    #         trajs_len.append(traj_len)
    #     self.traj_len = torch.Tensor(trajs_len)
    #
    # def get_trajs(self, trajs_idx):
    #     if trajs_idx is None:
    #         trajs = self.traj_data
    #         lens = self.traj_len
    #     elif type(trajs_idx) == list:
    #         trajs = self.traj_data[trajs_idx]
    #         lens = self.traj_len[trajs_idx]
    #     elif type(trajs_idx) == tuple:
    #         trajs = self.traj_data[trajs_idx[0]:trajs_idx[1]]
    #         lens = self.traj_len[trajs_idx[0]:trajs_idx[1]]
    #     elif type(trajs_idx) == int:
    #         trajs = self.traj_data[trajs_idx]
    #         lens = self.traj_len[trajs_idx]
    #     else:
    #         raise Exception("trajs_idx type error")
    #     mask = torch.ones(trajs.size(0), torch.max(lens, dim=-1), dtype=torch.bool, )
    #
    #     for i, length in enumerate(lens):
    #         mask[i, :length] = False
    #
    #     return trajs, mask.to(self.device)

    # def store_neutraj_train_data(self, **kwargs):
    #
    #     anchor_input, trajs_input, negative_input = kwargs['trajs'][0], kwargs['trajs'][1], kwargs['trajs'][2]
    #     anchor_input, trajs_input, negative_input = torch.Tensor(anchor_input).to(self.device), torch.Tensor(
    #         trajs_input).to(self.device), torch.Tensor(negative_input).to(self.device)
    #
    #     anchor_input_len, trajs_input_len, negative_input_len = kwargs['trajs_len'][0], kwargs['trajs_len'][1], \
    #     kwargs['trajs_len'][2]
    #     anchor_mask = torch.ones(anchor_input.size(0), anchor_input.size(1), dtype=torch.bool,
    #                              device=anchor_input.device)
    #     trajs_mask = torch.ones(trajs_input.size(0), trajs_input.size(1), dtype=torch.bool, device=trajs_input.device)
    #     negative_mask = torch.ones(negative_input.size(0), negative_input.size(1), dtype=torch.bool,
    #                                device=negative_input.device)
    #     # 然后，根据每个序列的实际长度来更新 mask，使得每个序列的非padding部分为True
    #     for i, length in enumerate(anchor_input_len):
    #         anchor_mask[i, :length] = False
    #     for i, length in enumerate(trajs_input_len):
    #         trajs_mask[i, :length] = False
    #     for i, length in enumerate(negative_input_len):
    #         negative_mask[i, :length] = False
    #     self.tmp_data = {'anchor': anchor_input, 'trajs': trajs_input, 'negative': negative_input,
    #                      'anchor_mask': anchor_mask, 'trajs_mask': trajs_mask, 'negative_mask': negative_mask}
    #
    # def store_neutraj_valid_data(self, **kwargs):
    #     pass


def cal_top10_acc(ground_truth, traj_embeddings, _range, lorenz: Lorenz, config_lorenz=None, quick=False,
                  _10in50=False):
    with torch.no_grad():
        ground_truth = ground_truth[_range[0]:_range[-1] + 1, _range[0]:_range[-1] + 1]
        traj_embeddings = traj_embeddings[_range[0]:_range[-1] + 1]
        fake_metrix = torch.zeros((ground_truth.shape[0], ground_truth.shape[1]))
        if type(traj_embeddings) != torch.Tensor:
            traj_embeddings = torch.tensor(traj_embeddings)
        traj_embeddings = traj_embeddings.detach()
        # Pre-generating lorenz ration
        lorenz.gen_valid_ratio_emb((_range[0], _range[-1]))

        for i in range(len(ground_truth)):
            traj_e = traj_embeddings[i].repeat(len(traj_embeddings), 1)
            # if lorenz.lorenz == 0 or (config_lorenz is not None and config_lorenz == 0) or config.lorenz == 0:
            if lorenz.lorenz == 0 or (config_lorenz is not None and config_lorenz == 0):
                fake_metrix[i, :] = torch.sum(torch.square(traj_e - traj_embeddings), dim=-1)
                # fake_metrix[i, :] = lorenz.dist(traj_e, traj_embeddings, config_lorenz)
            else:
                list_i, list_j = [i + _range[0] for _ in range(_range[0], _range[-1])], list(
                    range(_range[0], _range[-1]))
                # lorenz embedding
                fake_metrix[i, :] = lorenz.learned_cmb_dist(traj_e, traj_embeddings, list_i, list_j, train=False)
        fake_metrix = fake_metrix.cpu().detach().numpy()
        acc_10 = calculate_top_K_accuracy_for_whole_valid_traj(ground_truth, fake_metrix)
        if quick:
            if _10in50:
                return acc_10, 0., 0., 0., 0., (ground_truth, fake_metrix)
            return acc_10, 0., 0., 0., (ground_truth, fake_metrix)
        acc_50 = calculate_top_K_accuracy_for_whole_valid_traj(ground_truth, fake_metrix, tops=50, topf=50)
        ndcg_100 = ndcg(ground_truth, fake_metrix, 100)
        acc5 = calculate_top_K_accuracy_for_whole_valid_traj(ground_truth, fake_metrix, tops=5, topf=5)
        if _10in50:
            _10in50 = calculate_top_K_accuracy_for_whole_valid_traj(ground_truth, fake_metrix, tops=10, topf=50)
            return acc_10, acc_50, acc5, ndcg_100, _10in50, (ground_truth, fake_metrix)
    return acc_10, acc_50, acc5, ndcg_100, (ground_truth, fake_metrix)


def cal_top10_acc_bak(ground_truth, traj_embeddings, _range, lorenz: Lorenz, config_lorenz=None, quick=False,
                      _10in50=False):
    with torch.no_grad():
        ground_truth = ground_truth[_range[0]:_range[-1] + 1, _range[0]:_range[-1] + 1]
        traj_embeddings = traj_embeddings[_range[0]:_range[-1] + 1]
        fake_metrix = torch.zeros((ground_truth.shape[0], ground_truth.shape[1]))
        if type(traj_embeddings) != torch.Tensor:
            traj_embeddings = torch.tensor(traj_embeddings)
        traj_embeddings = traj_embeddings.detach()
        # Pre-generating lorenz ration
        lorenz.gen_valid_ratio_emb((_range[0], _range[-1]))
        for i in range(len(ground_truth)):
            traj_e = traj_embeddings[i].repeat(len(traj_embeddings), 1)
            # if lorenz.lorenz == 0 or (config_lorenz is not None and config_lorenz == 0) or config.lorenz == 0:
            if lorenz.lorenz == 0 or (config_lorenz is not None and config_lorenz == 0):
                fake_metrix[i, :] = torch.sum(torch.square(traj_e - traj_embeddings), dim=-1)
                # fake_metrix[i, :] = lorenz.dist(traj_e, traj_embeddings, config_lorenz)
            else:
                list_i, list_j = [i + _range[0] for _ in range(_range[0], _range[-1])], list(
                    range(_range[0], _range[-1]))
                # lorenz embedding
                fake_metrix[i, :] = lorenz.learned_cmb_dist(traj_e, traj_embeddings, list_i, list_j, train=False)
        return
        fake_metrix = fake_metrix.cpu().detach().numpy()

        # ndcg_100 = ndcg(ground_truth, fake_metrix, 100)
        acc_10 = calculate_top_K_accuracy_for_whole_valid_traj(ground_truth, fake_metrix)
        if quick:
            if _10in50:
                return acc_10, 0., 0., 0., 0., (ground_truth, fake_metrix)
            return acc_10, 0., 0., 0., (ground_truth, fake_metrix)
        acc_50 = calculate_top_K_accuracy_for_whole_valid_traj(ground_truth, fake_metrix, tops=50, topf=50)

        acc5 = calculate_top_K_accuracy_for_whole_valid_traj(ground_truth, fake_metrix, tops=5, topf=5)
        if _10in50:
            _10in50 = calculate_top_K_accuracy_for_whole_valid_traj(ground_truth, fake_metrix, tops=10, topf=50)
            return acc_10, acc_50, acc5, np.nan, _10in50, (ground_truth, fake_metrix)
    return acc_10, acc_50, acc5, np.nan, (ground_truth, fake_metrix)


def ndcg(ground_truth, fake_metrix, topk=10):
    assert isinstance(ground_truth, np.ndarray)
    assert isinstance(fake_metrix, np.ndarray)
    ndcg_value = ndcg_at_k(np.exp(-ground_truth), np.exp(-fake_metrix), topk)
    return ndcg_value


def calculate_top_K_accuracy_for_whole_valid_traj(sim_valid, fake_metrix, tops=10, topf=10):
    if len(sim_valid) != fake_metrix.shape[0] or len(sim_valid) != fake_metrix.shape[1]:
        raise Exception("similar metrix shape dismatch")
    if max(tops, topf) >= fake_metrix.shape[0]:
        raise Exception("too less traj(", fake_metrix.shape[0], ")for top", tops, topf)

    sum_acc = 0.0

    for i in range(len(sim_valid)):
        topA = calculate_top_K_min2(sim_valid[i], tops, skip=i, fake=False)
        topB = calculate_top_K_min2(fake_metrix[i], topf, skip=i)
        # if not (topf == len(topB) and len(topA) == tops):
        #     raise Exception(f"error topA B len,{len(topA)},{len(topB)}")
        common = [v for v in topA if v in topB]
        acc = float(len(common)) / tops
        sum_acc = sum_acc + acc
    return sum_acc / fake_metrix.shape[0]


def calculate_top_K_min2(dist, topk, skip, fake=True):
    dist_copy = np.copy(dist)
    dist_copy[skip] = np.Inf
    sorted_indices = np.argsort(dist_copy)
    ret = sorted_indices[:topk].tolist()
    if not fake:
        min_val = dist_copy[sorted_indices[topk - 1]]
        additional_indices = np.where(dist_copy == min_val)[0]
        for idx in additional_indices:
            if idx not in ret:
                ret.append(idx)
    return ret


def calculate_top_K_min(dist, topk, skip, fake=True):
    ret = []
    dist = copy.deepcopy(dist)
    dist[skip] = np.Inf
    for i in range(topk):
        min = np.Inf
        cnt = 0
        pos = -1
        for j in range(len(dist)):
            if dist[j] < min:
                min = dist[j]
                pos = j
                cnt = 1
            elif dist[j] == min:
                cnt += 1
        if pos == -1:
            raise Exception("..." + str(dist))
            # exit(123)
        else:
            ret.append(pos)
            dist[pos] = np.Inf
    if not fake:
        for j in range(len(dist)):
            if dist[j] == min:
                ret.append(j)
                cnt -= 1
        assert cnt == 1
    return ret


def ndcg_at_k(y_true, y_pred, k=5):
    r"""
        Evaluation function of NDCG@K
    """
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert len(y_pred.shape) == 2 and len(y_pred.shape) == 2

    num_of_users, num_pos_items = y_true.shape
    sorted_ratings = -np.sort(-y_true)  # descending order !!
    discounters = np.tile([np.log2(i + 1) for i in range(1, 1 + num_pos_items)], (num_of_users, 1))
    normalizer_mat = (np.exp2(sorted_ratings) - 1) / discounters

    sort_idx = (-y_pred).argsort(axis=1)  # index of sorted predictions (max->min)
    gt_rank = np.array([np.argwhere(sort_idx == i)[:, 1] + 1 for i in
                        range(num_pos_items)]).T  # rank of the ground-truth (start from 1)
    hit = (gt_rank <= k)

    # calculate the normalizer first
    normalizer = np.sum(normalizer_mat[:, :k], axis=1)
    # calculate DCG
    DCG = np.sum(((np.exp2(y_true) - 1) / np.log2(gt_rank + 1)) * hit.astype(float), axis=1)
    return np.mean(DCG / normalizer)


def ndcg_at_k_asc(y_true, y_pred, k=5):
    assert isinstance(y_pred, np.ndarray)
    assert isinstance(y_true, np.ndarray)
    assert len(y_pred.shape) == 2 and len(y_true.shape) == 2

    num_of_users, num_pos_items = y_true.shape

    # 计算理想情况下的 DCG（iDCG）
    sorted_true_ratings = np.sort(y_true, axis=1)[:, :k]  # 升序排列并截取前k个元素
    discounters = np.log2(np.arange(2, k + 2))
    ideal_dcg = np.sum(sorted_true_ratings / discounters, axis=1)

    # 根据 y_pred 的升序排列计算 DCG
    sorted_idx = np.argsort(y_pred, axis=1)[:, :k]  # 升序排列并截取前k个元素
    dcg = np.zeros(num_of_users)
    for i in range(num_of_users):
        dcg[i] = sum(y_true[i, sorted_idx[i]] / discounters)

    # 计算 NDCG
    ndcg = np.mean(dcg / ideal_dcg)

    return ndcg

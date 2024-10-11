import time

import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, lorentz):
        super(Loss, self).__init__()
        self.subloss = SubTrajLoss(lorentz)
        self.matchloss = MatchingLoss(lorentz)
        self.lorentz = lorentz

    def forward(self, xi, xj, sub_idx, x_i, x_j, label, match_idx, weight, idx_i, idx_j):
        loss1 = self.subloss(xi, xj, sub_idx, label, weight, idx_i, idx_j)
        loss2 = self.matchloss(x_i, x_j, match_idx, weight, idx_i, idx_j)
        return loss1
        # return loss1 + loss2


class SubTrajLoss(nn.Module):
    def __init__(self, lorentz):
        super(SubTrajLoss, self).__init__()
        self.lorentz = lorentz

    def forward(self, xi, xj, idx, label, weight, _idx_i, _idx_j):
        # N * r * (len1, len2) ---idx
        idx_i, idx_j = idx[:, :, 0], idx[:, :, 1]
        # N * r --- label
        N = xi.shape[0]
        # vi = torch.stack([x[n, idx_i[n], :] for n in range(N)])
        #
        # vj = torch.stack([x[n, idx_j[n], :] for n in range(N)])

        vi_ = xi[torch.arange(N)[:, None], idx_i]
        vj_ = xj[torch.arange(N)[:, None], idx_j]
        if self.lorentz.lorentz == 0:
            dist = torch.exp(-((vi_ - vj_) ** 2).sum(dim=-1))
        else:
            tmp_id_i = [item for item in _idx_i for _ in range(11)]
            tmp_id_j = [item for item in _idx_j for _ in range(11)]

            dist = torch.exp(
                -self.lorentz.learned_cmb_dist(vi_.view(-1, vi_.shape[-1]), vj_.view(-1, vj_.shape[-1]),
                                              tmp_id_i, tmp_id_j, default=False))
            # dist = torch.exp(-self.lorentz.learned_cmb_dist(vi_.view(-1, vi_.shape[-1]), vj_.view(-1, vj_.shape[-1]), default=True))

            dist = dist.view(vi_.shape[0], vi_.shape[1])

        # loss = ((label - dist) ** 2).sum() / 10
        label_ = torch.exp(-label)
        loss = (((label_ - dist) ** 2).sum(dim=-1) * weight / 10).sum()
        vi, vj = vi_[:, 0], vj_[:, 0]
        label2 = label[:, 0]

        #-------------- using learned_cmb_dist
        ###############
        if self.lorentz.lorentz == 0:
            dist = torch.exp(-((vi - vj) ** 2).sum(dim=-1))
        else:
            dist = torch.exp(-self.lorentz.learned_cmb_dist(vi, vj, _idx_i, _idx_j))
        ###############
        #--------------

        label2_ = torch.exp(-label2)
        # loss += ((label2 - dist) ** 2).sum()
        loss += (((label2_ - dist) ** 2) * weight).sum()
        return loss


class MatchingLoss(nn.Module):
    def __init__(self, lorentz):
        super(MatchingLoss, self).__init__()
        self.lorentz = lorentz

    def forward(self, xi, xj, idx, weight, _idx_i, _idx_j):
        idx_i, idx_j, idx__j = idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]
        N = xi.shape[0]


        vi_ = xi[torch.arange(N)[:, None], idx_i]
        vj_ = xj[torch.arange(N)[:, None], idx_j]
        v_j_ = xj[torch.arange(N)[:, None], idx__j]
        _idx__j = _idx_j[torch.arange(N)[:, None], idx__j]
        if self.lorentz.lorentz == 0:
            dist = torch.exp(-((vi_ - vj_) ** 2).sum(dim=-1))
            dist2 = torch.exp(-((vi_ - v_j_) ** 2).sum(dim=-1))
        else:
            dist = torch.exp(-self.lorentz.learned_cmb_dist(vi_, vj_, _idx_i, _idx_j)).sum(dim=-1)
            dist2 = torch.exp(-self.lorentz.learned_cmb_dist(vi_, v_j_, _idx_i, _idx__j)).sum(dim=-1)
            #dist = torch.exp(-self.lorentz.learned_cmb_dist(vi, vj, _idx_i, _idx_j))
        # dist = torch.exp(-((vi_ - vj_) ** 2).sum(dim=-1))
        # dist2 = torch.exp(-((vi_ - v_j_) ** 2).sum(dim=-1))
        loss = -torch.relu(dist - dist2 - 0.01).sum(dim=-1) * weight / 10
        # loss = min(0, 0.01 - dist + dist2).sum() / 10
        return loss.sum()


if __name__ == '__main__':
    N, L, H, r = 23, 31, 41, 10  # Re-defining dimensions for clarity
    tensor = torch.tensor([i for i in range(N * L * H)], dtype=torch.float32).resize(N, L, H)
    idx = torch.randint(0, L, (N, r, 2))
    model = SubTrajLoss()
    label = torch.randn(N, r)
    output = model(tensor, tensor, idx, label)
    print(output)
    model = MatchingLoss()
    idx = torch.randint(0, L, (N, r, 3))
    output = model(tensor, tensor, idx)
    print(output)

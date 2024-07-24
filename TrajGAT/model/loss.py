import torch
from torch.nn import Module, PairwiseDistance


class WeightedRankingLoss(Module):
    def __init__(self, sample_num, alpha, device, lorenz=None):
        super(WeightedRankingLoss, self).__init__()
        self.alpha = alpha
        weight = [1]
        single_sample = sample_num // 2

        for index in range(single_sample, 1, -1):
            weight.append(index)
        for index in range(1, single_sample + 1, 1):
            weight.append(index)

        # 权重归一化
        weight = torch.tensor(weight, dtype=torch.float)
        self.weight = (weight / torch.sum(weight)).to(device)
        self.lorenz = lorenz

    def forward(self, vec, all_dis, id_list=None):
        """
        vec [batch_size, sample_num, d_model]
        dis [batch_size, sample_num]
        """
        all_loss = 0
        batch_num = vec.size(0)
        sample_num = vec.size(1)

        for batch in range(batch_num):
            traj_list = vec[batch]
            dis_list = all_dis[batch]
            idxs = id_list[batch]

            anchor_trajs = traj_list[0].repeat(sample_num, 1)
            anchor_idxs = idxs[0].repeat(sample_num, 1).view(-1)
            assert self.lorenz is not None
            if self.lorenz.lorenz == 0:
                pairdist = PairwiseDistance(p=2)
                dis_pred = pairdist(anchor_trajs, traj_list)
                # [sample_num]
                # dis_pred = self.lorenz.dist(anchor_trajs, traj_list)
                # dis_pred = self.lorenz.learned_cmb_dist(anchor_trajs, traj_list, traj_i=None, traj_j=None)
                sim_pred = torch.exp(-dis_pred)

            else:
                # dis_pred = self.lorenz.dist(anchor_trajs, traj_list)
                dis_pred = self.lorenz.learned_cmb_dist(anchor_trajs, traj_list, traj_i=anchor_idxs.tolist(), traj_j=idxs.tolist())
                sim_pred = torch.exp(-dis_pred)

            sim_truth = torch.exp(-self.alpha * dis_list)
            div = sim_truth - sim_pred
            square = torch.mul(div, div)
            weighted_square = torch.mul(square, self.weight)
            loss = torch.sum(weighted_square)

            all_loss = all_loss + loss

        return all_loss

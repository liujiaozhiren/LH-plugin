import os
import statistics

import numpy as np
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


class Seq2VecTrans(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(Seq2VecTrans, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        self.linear_ = nn.Linear(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, mask=None):
        x = self.linear_(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.linear(x[:, 0, :])
        return x


class Seq2VecLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(Seq2VecLSTM, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear_ = nn.Linear(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, lens=None):
        x = self.linear_(x)
        x, _ = self.rnn(x)
        # selected_vectors = tensor[torch.arange(N), idx]
        # vi = torch.stack([tensor[n, idx[n], :] for n in range(N)])
        if lens is not None:
            y = x[torch.arange(x.size(0)), lens - 1]
        else:
            y = x[:, -1, :]
        x = self.linear(y)
        return x


class Traj2Vector(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, trajs=None, model_type='lstm', device=None,
                 pretrain=False, init=None, load=None):
        super(Traj2Vector, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.type = model_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.valid_ratio = None


        # transformer_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
        # self.transformer = torch.nn.TransformerEncoder(transformer_layer, num_layers=2)
        # self.linear_ = torch.nn.Linear(input_size, hidden_size)
        # self.linear = torch.nn.Linear(hidden_size, 12)
        if self.type == 'lstm':
            self.seq2vec = Seq2VecLSTM(input_size, hidden_size, 12, num_layers)
            if load is not None and os.path.exists(
                    f'/lorentz_{model_type}_{load[0]}_{load[1]}_0.0642_best.pth'):
                # f"lorentz_{model_type}_{city}_{sim}_{best_ratio:.4f}_best.pth"
                # self.load_state_dict(torch.load(f'../data_set/{load[0]}/model_{load[1]}_best.pth'))
                self.load_state_dict(torch.load(f'/lorentz_{model_type}_{load[0]}_{load[1]}_0.0642_best.pth'))
                print('load pre train lorentz model:',
                      f'../lorentz/lorentz_{model_type}_{load[0]}_{load[1]}_0.0642_best.pth')
            else:
                print('not load pre train lorentz model:', f'../lorentz/lorentz_{model_type}_best.pth')
        elif self.type == 'transformer':
            self.seq2vec = Seq2VecTrans(input_size, hidden_size, 12, num_layers)
            if load is not None and os.path.exists(
                    f'/lorentz_{model_type}_{load[0]}_{load[1]}_0.0642_best.pth'):
                self.load_state_dict(torch.load(f'/lorentz_{model_type}_{load[0]}_{load[1]}_0.0642_best.pth'))

                print('load pre train lorentz model:',
                      f'../lorentz/lorentz_{model_type}_{load[0]}_{load[1]}_0.0642_best.pth')
            else:
                print('not load pre train lorentz model:', f'../lorentz/lorentz_{model_type}_best.pth')
        else:
            raise Exception("type not support")
        self.init_params()
        if trajs is not None:
            self.store_traj_data(trajs)

        self.to(self.device)
        if init is not None:
            self.model_init_pretrain(init)

    def model_init_pretrain(self, ratio0=1):
        idx_list = [[i, j] for i in range(10000) for j in range(10000) if i!=j]
        data = np.array(idx_list)
        data = DataLoader(data, batch_size=10000, shuffle=True)
        opt = Adam(self.seq2vec.parameters(), lr=1e-5)
        for epoch in range(50):
            sum_loss = []
            with tqdm(data, total=len(data)) as t:
                for _i, idx in enumerate(t):
                    idx_i, idx_j = idx[:, 0], idx[:, 1]
                    ratio_0, _, _ = self.gen_ration(idx_i.tolist(), idx_j.tolist(), 2)
                    loss = torch.mean((ratio_0 - ratio0) ** 2)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    sum_loss.append(loss.item())
                    a = statistics.mean(sum_loss[:2000])
                    t.set_postfix(loss=a)
                    if a < 1e-3 and _i > 1000:
                        print(f"the initial loss < 1e-4:{a}")
                        return
            print("epoch:", epoch, "loss:", statistics.mean(sum_loss))

    def init_params(self):
        for param in self.parameters():
            if len(param.size()) == 2:
                torch.nn.init.xavier_normal_(param)
            else:
                torch.nn.init.zeros_(param)

    # def forward(self, x, mask=None):
    #     x = self.linear_(x)
    #     x = self.transformer(x, src_key_padding_mask=mask)
    #     x = self.linear(x[:, 0, :])
    #     return x

    def store_traj_data(self, trajs, need_norm=False):

        assert len(trajs) >= 10000
        # self.traj_data = torch.Tensor(trajs).to(self.device)

        if need_norm:
            x, y = [], []
            for traj in trajs:
                for point in traj:
                    x.append(point[0])
                    y.append(point[1])
            x = np.array(x)
            y = np.array(y)
            mean_x, mean_y = x.mean(), y.mean()
            std_x, std_y = x.std(), y.std()
            for traj in trajs:
                for i, point in enumerate(traj):
                    point0 = (point[0] - mean_x) / std_x
                    point1 = (point[1] - mean_y) / std_y
                    traj[i] = (point0, point1)

        trajs_len = []
        for traj in trajs:
            traj_len = len(traj)
            trajs_len.append(traj_len)
        self.traj_data = torch.zeros(len(trajs), max(trajs_len), 2, dtype=torch.float32, device=self.device)
        self.max_traj_len = max(trajs_len)
        for i, traj in enumerate(trajs):
            self.traj_data[i, :len(traj)] = torch.Tensor(traj)
        self.traj_len = torch.tensor(trajs_len, dtype=torch.long)

    def dynamic_store_traj_data(self, trajs, dim=None, lens=None):
        if dim is None:
            dim = len(trajs[0][0])
        trajs_len = []
        if lens is None:
            for traj in trajs:
                traj_len = len(traj)
                trajs_len.append(traj_len)
            self.traj_data = torch.zeros(len(trajs), max(trajs_len), dim, dtype=torch.float32, device=self.device)
            self.max_traj_len = max(trajs_len)
            for i, traj in enumerate(trajs):
                self.traj_data[i, :len(traj)] = torch.Tensor(traj)
            self.traj_len = torch.tensor(trajs_len, dtype=torch.long)
        else:
            self.traj_data = trajs
            self.traj_len = lens
            self.max_traj_len = max(lens)

    def get_trajs(self, trajs_idx):
        if trajs_idx is None:
            trajs = self.traj_data
            lens = self.traj_len
        elif type(trajs_idx) == list:
            trajs = self.traj_data[trajs_idx]
            lens = self.traj_len[trajs_idx]
        elif type(trajs_idx) == tuple:
            trajs = self.traj_data[trajs_idx[0]:trajs_idx[1]]
            lens = self.traj_len[trajs_idx[0]:trajs_idx[1]]
        elif type(trajs_idx) == int or type(trajs_idx) == np.int64:
            trajs = self.traj_data[trajs_idx:trajs_idx + 1]
            lens = self.traj_len[trajs_idx:trajs_idx + 1]
        else:
            raise Exception("trajs_idx type error")
        mask = torch.ones(trajs.size(0), self.max_traj_len, dtype=torch.bool, )

        for i, length in enumerate(lens):
            mask[i, :length.item()] = False

        return trajs.to(self.device), mask.to(self.device), lens.to(self.device), torch.max(lens).item()

    def gen_ration(self, traj_i, traj_j, type=2):
        trajs_i, mask_i, len_i, max_l_i = self.get_trajs(traj_i)
        trajs_j, mask_j, len_j, max_l_j = self.get_trajs(traj_j)
        l = max(max_l_i, max_l_j)

        # emb_i = self(trajs_i[:, :l], mask_i[:, :l])
        # emb_j = self(trajs_j[:, :l], mask_j[:, :l])
        if self.type == 'transformer':
            emb_i = self.seq2vec(trajs_i[:, :l], mask_i[:, :l])
            emb_j = self.seq2vec(trajs_j[:, :l], mask_j[:, :l])
        elif self.type == 'lstm':
            emb_i = self.seq2vec(trajs_i[:, :l], len_i)
            emb_j = self.seq2vec(trajs_j[:, :l], len_j)

        # rst = emb_i * emb_j
        # rst = torch.softmax(rst, dim=1)
        # return rst[:, 0], rst[:, 1], rst[:, 2]
        return self.emb2ratio(emb_i * emb_j, type)
        # pass

    def emb2ratio(self, emb, ratio_type=3):
        if ratio_type == 3:
            assert emb.shape[1] % 3 == 0
            k = emb.shape[1] // 3
            sum1 = emb[:, 0:k].sum(dim=1)
            sum2 = emb[:, k:2 * k].sum(dim=1)
            sum3 = emb[:, 2 * k:3 * k].sum(dim=1)

            # Concatenating the sums to form a [n, 3] vector
            result = torch.stack([sum1, sum2, sum3], dim=1)

            # Applying softmax on the final [n, 3] vector
            sr = torch.softmax(result, dim=1)

            return sr[:, 0], sr[:, 1], sr[:, 2]
        elif ratio_type == 2:
            assert emb.shape[1] % 2 == 0
            k = emb.shape[1] // 2
            sum1 = emb[:, 0:k].sum(dim=1)
            sum2 = emb[:, k:2 * k].sum(dim=1)
            result = torch.stack([sum1, sum2], dim=1)
            sr = torch.softmax(result, dim=1)
            return sr[:, 0], None, sr[:, 1]
        else:
            raise Exception("type error")

    def gen_valid_ratio_emb(self, _range):
        trajs, masks, lens, max_l = self.get_trajs(_range)
        if self.type == 'transformer':
            emb = self.seq2vec(trajs, masks).to('cpu')
        elif self.type == 'lstm':
            emb = self.seq2vec(trajs, lens).to('cpu')
        if self.valid_ratio is None:
            self.valid_ratio = torch.zeros((_range[-1], emb.shape[1]), dtype=torch.float32, device='cpu')
        self.valid_ratio[_range[0]:_range[1]] = emb
        self.valid_ratio.detach()

    def get_ration(self, traj_i, traj_j, ratio_type=2):
        if traj_i is None:
            emb_i = self.valid_ratio
        elif type(traj_i) == list:
            emb_i = self.valid_ratio[traj_i]
        elif type(traj_i) == tuple:
            emb_i = self.valid_ratio[traj_i[0]:traj_i[1]]
        elif type(traj_i) == int:
            emb_i = self.valid_ratio[traj_i:traj_i + 1]
        else:
            raise Exception("trajs_idx type error")

        if traj_j is None:
            emb_j = self.valid_ratio
        elif type(traj_j) == list:
            emb_j = self.valid_ratio[traj_j]
        elif type(traj_j) == tuple:
            emb_j = self.valid_ratio[traj_j[0]:traj_j[1]]
        elif type(traj_j) == int:
            emb_j = self.valid_ratio[traj_j:traj_j + 1]
        else:
            raise Exception("trajs_idx type error")

        assert emb_i.shape[0] == emb_j.shape[0]

        # rst = emb_i * emb_j
        # return rst[:, 0], rst[:, 1], rst[:, 2]
        return self.emb2ratio(emb_i * emb_j, ratio_type)

    # def train(self, data):
    #     trajs, trajs_len, labels = data
    #     mask = torch.ones(trajs.size(0), trajs.size(1), dtype=torch.bool,device=trajs.device)
    #
    #     # 然后，根据每个序列的实际长度来更新 mask，使得每个序列的非padding部分为True
    #     for i, length in enumerate(trajs_len):
    #         mask[i, :length] = False
    #
    #     out = self(trajs, mask)
    #     pass


if __name__ == '__main__':
    N, L, H = 2, 3, 4  # Example dimensions
    tensor = torch.tensor([i for i in range(N * L * H)], dtype=torch.float32).resize(N, L, H)
    idx = torch.randint(0, L, (N,)).view(-1)

    # tensor = torch.randn(N, L, H)  # Tensor of shape [N, L, H]
    # idx = torch.randint(0, L, (N,))  # Index tensor of shape [N]
    print(tensor, idx)
    # Selecting the H-dimensional vectors from each N according to idx
    selected_vectors = tensor[torch.arange(N), idx]
    vi = torch.stack([tensor[n, idx[n], :] for n in range(N)])
    print(selected_vectors, vi)

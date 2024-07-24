import pickle
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lorenz.lorenz_net import Traj2Vector
from lorenz.transfer import Lorenz

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trajs, triangle = None, None


def pdump(file_content, file_path):
    with open(file_path, "wb") as tar:
        pickle.dump(file_content, tar)


def pload(file_path):
    with open(file_path, "rb") as tar:
        out = pickle.load(tar)
    return out


def InfoNce(embedding):
    embedding = embedding.view(-1, 3)
    traj_1, traj_2, traj_3 = embedding[:, 0], embedding[:, 1], embedding[:, 2]


def loader_collate_fn(batch):
    uandv = []
    used_i, used_j, used_k = [], [], []

    for i in batch:
        i = i % 10000
        j, k = None, None
        u, v = None, None
        while True:
            sample = random.sample(range(len(trajs)), 2)
            j, k = sample[0], sample[1]
            if i == j or i == k:
                continue
            u, v = triangle[i][j], triangle[i][k]
            if u[0] > v[0]:
                j, k = k, j
            u, v = triangle[i][j], triangle[i][k]
            if u[1] <= v[1]:
                break
        used_i.append(i)
        used_j.append(j)
        used_k.append(k)
        uandv.append([u, v])
    uandv = torch.tensor(uandv, dtype=torch.float32, device=device)
    traj_idx = (used_i, used_j, used_k)
    return traj_idx, uandv


def preproc(max_value):
    return list(range(max_value))


def train(city='chengdu', sim='dtw', model_type='lstm'):
    global trajs
    trajs = pload(f'../data_set/{city}/trajs_10000.pkl')

    global triangle
    triangle = pload(f'../data_set/{city}/tri_ineq_{city}_{sim}_10000x10000.pkl')

    model = Traj2Vector(2, 96 // 4, trajs=None, device=device, model_type=model_type, pretrain=True).to(device)
    # lorenz = Lorenz(base_model=[], dim=(2, 128), lorenz=0, trajs=trajs)
    model.store_traj_data(trajs, need_norm=True)
    model.train()
    model_optim = torch.optim.SGD(model.parameters(), lr=1e-5)

    criterion = nn.MarginRankingLoss(margin=0.1)
    data = preproc(100000)
    # data_valid = preproc_valid(trajs)
    import gc
    subdist = None  # 或 del subdist
    gc.collect()
    dataloader = DataLoader(data, batch_size=1000, shuffle=True, collate_fn=loader_collate_fn)
    validloader = DataLoader(preproc(200000), batch_size=1000, shuffle=False, collate_fn=loader_collate_fn)
    # validloader = DataLoader(data_valid, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn_valid)
    bst_loss = 1e9
    best_ratio = -1
    # dist = np.array(dist)
    # lam = 0.5
    early_stop = 0
    for epoch in range(500):
        model.train()
        epoch_loss = []
        model.train()
        with tqdm(dataloader) as tq:
            for traj_idx, u_v in tq:
                traj_idx_i, traj_idx_j, traj_idx_k = traj_idx
                near = model.gen_ration(traj_idx_i, traj_idx_j, 2)[2]
                far = model.gen_ration(traj_idx_i, traj_idx_k, 2)[2]
                # loss = criterion(far, near, torch.ones_like(far))
                u, v = u_v[:, 0, 0], u_v[:, 1, 0]

                assert (u <= v).all()
                # label = v - u
                # input = far - near

                # loss1 = torch.nn.functional.mse_loss(far, v) + torch.nn.functional.mse_loss(near, u)
                # combined_var = torch.stack([input, label], dim=0)
                # correlation_matrix = torch.corrcoef(combined_var)
                # loss2 = -correlation_matrix[0, 1]
                # loss = loss1 + loss2
                loss = torch.nn.functional.mse_loss(far, v) + torch.nn.functional.mse_loss(near, u)
                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

                tq.set_postfix(loss=loss)

        # epoch_loss = torch.tensor(epoch_loss)
        # if epoch_loss.mean() > 0.5:
        #     lam = lam * 0.5
        #     print("lamba:", lam, "mean no loss", epoch_loss.mean())

        x, y = [], []
        with torch.no_grad():
            for traj_idx, u_v in validloader:
                traj_idx_i, traj_idx_j, traj_idx_k = traj_idx
                near = model.gen_ration(traj_idx_i, traj_idx_j, 2)[2]
                far = model.gen_ration(traj_idx_i, traj_idx_k, 2)[2]
                u, v = u_v[:, 0, 0], u_v[:, 1, 0]
                label = v - u
                input = far - near
                x.extend(far.tolist())
                x.extend(near.tolist())
                y.extend(v.tolist())
                y.extend(u.tolist())
        combined_var = torch.stack([torch.tensor(x), torch.tensor(y)], dim=0)
        correlation_matrix = torch.corrcoef(combined_var)
        ratio = correlation_matrix[0, 1]

        if ratio > best_ratio:
            best_ratio = ratio
            early_stop = 0
            torch.save(model.state_dict(), f'./lorenz_mod_file/lorenz_{model_type}_{city}_{sim}_{best_ratio:.4f}_best.pth')
            print(f'Best ratio :{ratio:.4f} save ./lorenz_{city}_{sim}_best.pth')
        else:
            early_stop += 1
            if early_stop > 6:
                break
            print(f'worse ratio :{ratio:.4f}')
        # bst5, bst50, bndcg, bst10in50 = max(bst5, hr5), max(bst50, hr50), max(bndcg, ndcg), max(bst10in50, hr10in50)
    # print(f'Best HR10:{bst10:.4f}, HR50:{bst50:.4f}, HR5:{bst5:.4f}, NDCG:{bndcg:.7f}, HR10in50:{bst10in50:.4f}')
    # model.load_state_dict(torch.load(f'../data_set/{city}/model_{sim}_best.pth'))
    #
    # emb = valid(validloader, model)
    # hr10, hr50, hr5, ndcg, hr10in50 = cal_top10_acc(dist, emb, (6000, 10000), lorenz, _10in50=True)
    #
    # print(f'Final HR10:{hr10:.4f}, HR50:{hr50:.4f}, HR5:{hr5:.4f}, NDCG:{ndcg:.7f}, HR10in50:{hr10in50:.4f}')


if __name__ == '__main__':
    train()

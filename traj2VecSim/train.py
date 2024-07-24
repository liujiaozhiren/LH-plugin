import pickle
import random
import argparse
import time

import numpy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('../')
from Traj2SimVec import cfg
from Traj2SimVec.loss import Loss
from Traj2SimVec.network import Traj2SimVec, Traj2SimVecHandler
from lorenz.transfer import cal_top10_acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pdump(file_content, file_path):
    with open(file_path, "wb") as tar:
        pickle.dump(file_content, tar)


def pload(file_path):
    with open(file_path, "rb") as tar:
        out = pickle.load(tar)
    return out


def preprocess(trajs, dist, subdist, np):
    lenth = int(len(trajs) * 0.6)
    near_sample_idx = []
    far_sample_idx = []
    for i in range(lenth):
        tmp_sim = []
        tmp_j = []
        for j in range(i, lenth):
            if np[i][j] is not None:
                tmp_sim.append(dist[i][j])
                tmp_j.append(j)
        # assert len(tmp_sim) >= 20
        kk = len(tmp_sim) // 2
        sorted_indices = sorted(range(len(tmp_sim)), key=lambda i: tmp_sim[i])
        # ready_sim = [tmp_sim[i] for i in sorted_indices]
        ready_j = [tmp_j[i] for i in sorted_indices]
        # near, far = ready_sim[:10], ready_sim[-10:]
        near_idx, far_idx = ready_j[:kk], ready_j[-kk:]
        for k in range(kk):
            near_sample_idx.append([i, near_idx[k], 10 - k])
            far_sample_idx.append([i, far_idx[k], 11 - kk + k])
    near_data_traj_i, far_data_traj_i = [], []
    near_data_traj_j, far_data_traj_j = [], []
    near_data_traj_i_idx, far_data_traj_i_idx = [], []
    near_data_traj_j_idx, far_data_traj_j_idx = [], []
    near_data_sim, far_data_sim = [], []
    near_data_sub_idx, far_data_sub_idx = [], []
    near_match_idx, far_match_idx = [], []
    near_weight, far_weight = [], []
    for item in near_sample_idx:
        i, j, weight = item[0], item[1], item[2]
        xi, xj = trajs[i], trajs[j]
        near_data_traj_i.append(xi)
        near_data_traj_j.append(xj)
        near_data_traj_i_idx.append(i)
        near_data_traj_j_idx.append(j)

        near_weight.append(weight)
        di = subdist[i][j]
        subsim, subidx = [], []
        assert (dist[i][j] - di[0][0]) ** 2 < 1e-3
        for item_ in di:
            sim, idx_i, idx_j = item_[0], int(item_[1]), int(item_[2])
            # float inf to max float32
            assert idx_i <= len(xi) and idx_j <= len(xj)
            if idx_i == 0:
                idx_i = 1
            if idx_j == 0:
                idx_j = 1
            if sim == float('inf'):
                sim = 1e10
            subsim.append(sim)
            subidx.append([idx_i - 1, idx_j - 1])
        near_data_sim.append(subsim)
        near_data_sub_idx.append(subidx)


        np_item = np[i][j]
        if len(np_item) < 10:
            max_traj_len = min(len(xi), len(xj))
            tmp_idx = random.sample(range(0, max_traj_len), 10)
            np_use = [[i,i] for i in tmp_idx]
            #np_use = np_item[random.choices(range(0, max_traj_len), k=10)]
        else:
            np_use = np_item[random.sample(range(len(np_item)), 10)]
        tmp_near_match_idx = []
        for item_ in np_use:
            u, v = item_[0], item_[1]
            assert u < len(trajs[i]) and v < len(trajs[j])
            flag = True
            while flag:
                q = random.randint(0, len(trajs[j]) - 1)
                flag = False
                for item__ in np_item:
                    u_, v_ = item__[0], item__[1]
                    if q == v_ and u == u_:
                        flag = True
                        break
            tmp_near_match_idx.append([u, v, q])
        near_match_idx.append(tmp_near_match_idx)

    for item in far_sample_idx:
        i, j, weight = item[0], item[1], item[2]
        xi, xj = trajs[i], trajs[j]
        far_data_traj_i.append(xi)
        far_data_traj_j.append(xj)
        far_data_traj_i_idx.append(i)
        far_data_traj_j_idx.append(j)
        far_weight.append(weight)
        di = subdist[i][j]
        subsim, subidx = [], []
        assert ((dist[i][j] - di[0][0])) ** 2 < 1e-3
        for item_ in di:
            sim, idx_i, idx_j = item_[0], int(item_[1]), int(item_[2])
            assert idx_i <= len(xi) and idx_j <= len(xj)
            if idx_i == 0:
                idx_i = 1
            if idx_j == 0:
                idx_j = 1
            if sim == float('inf'):
                sim = 1e10
            subsim.append(sim)
            subidx.append([idx_i - 1, idx_j - 1])
        far_data_sim.append(subsim)
        far_data_sub_idx.append(subidx)

        np_item = np[i][j]
        if len(np_item) < 10:
            max_traj_len = min(len(xi), len(xj))
            tmp_idx = random.sample(range(0, max_traj_len), 10)
            np_use = [[i, i] for i in tmp_idx]
            #np_use = np_item[random.choices(range(0, max_traj_len), k=10)]
        else:
            np_use = np_item[random.sample(range(len(np_item)), 10)]

        # np_use = np_item[random.sample(range(len(np_item)), 10)]
        tmp_far_match_idx = []
        for item_ in np_use:
            u, v = item_[0], item_[1]
            assert u < len(trajs[i]) and v < len(trajs[j])
            flag = True
            while flag:
                q = random.randint(0, len(trajs[j]) - 1)
                flag = False
                for item__ in np_item:
                    u_, v_ = item__[0], item__[1]
                    if q == v_ and u == u_:
                        flag = True
                        break
            tmp_far_match_idx.append([u, v, q])
        far_match_idx.append(tmp_far_match_idx)
    # near_traj_i, far_traj_i = torch.tensor(near_data_traj_i).to(device), torch.tensor(far_data_traj_i).to(device)
    # near_traj_j, far_traj_j = torch.tensor(near_data_traj_j).to(device), torch.tensor(far_data_traj_j).to(device)
    # near_sim, far_sim = torch.tensor(near_data_sim).to(device), torch.tensor(far_data_sim).to(device)
    # near_sub_idx, far_sub_idx = torch.tensor(near_data_sub_idx).to(device), torch.tensor(far_data_sub_idx).to(device)
    # near_match_idx, far_match_idx = torch.tensor(near_match_idx).to(device), torch.tensor(far_match_idx).to(device)
    # return (near_traj_i, near_traj_j, near_sim, near_sub_idx, near_match_idx), (
    #     far_traj_i, far_traj_j, far_sim, far_sub_idx, far_match_idx)
    assert len(near_data_traj_i) == len(far_data_traj_i)
    data = []
    for i in range(len(near_data_traj_i)):
        v = [near_data_traj_i[i], near_data_traj_j[i], near_data_sim[i], near_data_sub_idx[i], near_match_idx[i],
             far_data_traj_i[i], far_data_traj_j[i], far_data_sim[i], far_data_sub_idx[i], far_match_idx[i],
             near_weight[i], far_weight[i], near_data_traj_i_idx[i], near_data_traj_j_idx[i],
             far_data_traj_i_idx[i],far_data_traj_j_idx[i]]
        data.append(v)
    return data


def loader_collate_fn(batch):
    # B = len(batch)
    near_traj_i, near_traj_j, near_sim, near_sub_idx, near_match_idx = [], [], [], [], []
    far_traj_i, far_traj_j, far_sim, far_sub_idx, far_match_idx = [], [], [], [], []
    near_traj_i_idx, near_traj_j_idx, far_traj_i_idx, far_traj_j_idx = [], [], [], []
    max_near_len, max_far_len = 0, 0
    weight_near, weight_far = [], []
    for item in batch:
        near_traj_i.append(item[0])
        near_traj_j.append(item[1])
        max_near_len = max(max_near_len, len(item[0]))
        max_near_len = max(max_near_len, len(item[1]))
        near_sim.append(item[2])
        near_sub_idx.append(item[3])
        near_match_idx.append(item[4])
        far_traj_i.append(item[5])
        far_traj_j.append(item[6])
        max_far_len = max(max_far_len, len(item[5]))
        max_far_len = max(max_far_len, len(item[6]))
        far_sim.append(item[7])
        far_sub_idx.append(item[8])
        far_match_idx.append(item[9])
        a = item[9]
        b = item[4]
        aa_i = item[5]
        aa_j = item[6]
        bb_i = item[0]
        bb_j = item[1]
        for ite in a:
            u, v, q = ite[0], ite[1], ite[2]
            assert u < len(aa_i) and v < len(aa_j) and q < len(aa_j)
        for ite in b:
            u, v, q = ite[0], ite[1], ite[2]
            assert u < len(bb_i) and v < len(bb_j) and q < len(bb_j)
        weight_near.append(item[10])
        weight_far.append(item[11])
        near_traj_i_idx.append(item[12])
        near_traj_j_idx.append(item[13])
        far_traj_i_idx.append(item[14])
        far_traj_j_idx.append(item[15])


    near_traj_i.extend(near_traj_j)
    far_traj_i.extend(far_traj_j)
    near_padded_traj = [traj + [[0, 0]] * (max_near_len - len(traj)) for traj in near_traj_i]
    far_padded_traj = [traj + [[0, 0]] * (max_far_len - len(traj)) for traj in far_traj_i]
    near = (torch.tensor(near_padded_traj).to(torch.float32).to(device), torch.tensor(near_sim).to(device),
            torch.tensor(near_sub_idx).to(device), torch.tensor(near_match_idx).to(device),
            torch.tensor(weight_near).to(device), near_traj_i_idx, near_traj_j_idx)
    far = (torch.tensor(far_padded_traj).to(torch.float32).to(device), torch.tensor(far_sim).to(device),
           torch.tensor(far_sub_idx).to(device), torch.tensor(far_match_idx).to(device),
           torch.tensor(weight_far).to(device), far_traj_i_idx, far_traj_j_idx)
    return near, far


def preproc_valid(trajs):
    lenth = int(len(trajs) * 0.6)
    trajs = trajs[lenth:]
    lens = []
    for traj in trajs:
        lens.append(len(traj))
    max_len = max(lens)
    trajs = [traj + [[0, 0]] * (max_len - len(traj)) for traj in trajs]
    valid_data = []
    for i, traj in enumerate(trajs):
        valid_data.append([lens[i], traj])
    return valid_data


def collate_fn_valid(batch):
    B = len(batch)
    trajs, lens = [], []
    for item in batch:
        lens.append(item[0] - 1)
        trajs.append(item[1])

    return torch.tensor(trajs, dtype=torch.float32).to(device), torch.tensor(lens).to(device)


def valid(dataloader, model: Traj2SimVec):
    model.eval()
    embs = torch.zeros((len_traj, 128)).to(device)
    start = len_traj * cfg.train_ratio
    with torch.no_grad():
        for trajs, lens in dataloader:
            y, _ = model(trajs)
            emb = y[torch.arange(len(lens)), lens]
            embs[start:start + len(emb)] = emb
            start += len(emb)
    return embs


def data_enhance(trajs, dist, subdist):
    max_dist = dist.max()
    xs, ys = [], []

    for traj in trajs:
        for item in traj:
            x, y = item
            xs.append(x)
            ys.append(y)
    xs, ys = numpy.array(xs), numpy.array(ys)
    meanx, meany, stdx, stdy = xs.mean(), ys.mean(), xs.std(), ys.std()
    for traj in trajs:
        for idx in range(len(traj)):
            x, y = traj[idx]
            traj[idx] = [(x - meanx) / stdx, (y - meany) / stdy]

    max_dist = dist.max()
    dist[:] = dist[:]/max_dist
    subdist[:,:,:,0] = subdist[:,:,:,0]/max_dist

def train(load=False):

    trajs = pload(f'../data_set/{cfg.city}/trajs.pkl')

    dist = pload(f'../data_set/{cfg.city}/{cfg.sim}.pkl')
    dist = numpy.array(dist)
    np = pload(f'../data_set/{cfg.city}/{cfg.sim}_np.pkl')
    global len_traj
    len_traj = len(trajs)
    subdist = numpy.zeros((len_traj, len_traj, 11, 3), dtype=numpy.float32)
    # you should mod this subtraj sim matrix loading code.
    raise Exception("You should modify this code to load the subtraj sim matrix.")
    subdist = pload(f'../data_set/{city}/{sim}_sub.pkl')




    data_enhance(trajs, dist, subdist)
    data_ = preprocess(trajs, dist, subdist, np)
    data_ = data_[:int(len(data_) * cfg.ratio)]
    data_valid = preproc_valid(trajs)


    import gc
    subdist = None  # 或 del subdist
    gc.collect()
    dataloader = DataLoader(data_, batch_size=cfg.batch_size, shuffle=True, collate_fn=loader_collate_fn)
    validloader = DataLoader(data_valid, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn_valid)
    bst10, bst5, bst50, bndcg, bst10in50, early_stop = 0.0, 0.0, 0.0, 0.0, 0.0, 0

    model = Traj2SimVecHandler(cfg, trajs).to(device)
    # lorenz = Lorenz(base_model=[], dim=(2, 128), lorenz=cfg.lorenz, trajs=trajs, sqrt=cfg.sqrt)
    criterion = Loss(model.lorenz)
    model.train()
    model_optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    dist = numpy.array(dist)
    if load:
        model.load_state_dict(torch.load(f'../data_set/{cfg.city}/model_{cfg.sim}_best.pth'))
        emb = valid(validloader, model)
        bst10, bst50, bst5, bndcg, bst10in50,_ = cal_top10_acc(dist, emb, (int(cfg.train_ratio*len_traj), int(cfg.valid_ratio* len_traj)), model.lorenz, _10in50=True, quick=False)
    for epoch in range(500):
        if True:
            model.train()
            epoch_loss = 0.0
            model.train()
            model.lorenz.both_train(True, False)
            with tqdm(dataloader, 'raw') as tq:
                for near, far in tq:
                    near_traj, near_sim, near_sub_idx, near_match_idx, near_weight, n_idx_i, n_idx_j = near
                    far_traj, far_sim, far_sub_idx, far_match_idx, far_weight, f_idx_i, f_idx_j = far
                    near_sub, near_match = model(near_traj)  # vecters [B*SAM, d_model]
                    far_sub, far_match = model(far_traj)
                    half = near_sub.shape[0] // 2
                    loss_near = criterion(xi=near_sub[:half], xj=near_sub[half:], sub_idx=near_sub_idx,
                                          x_i=near_match[:half], x_j=near_match[half:], label=near_sim,
                                          match_idx=near_match_idx, weight=near_weight, idx_i=n_idx_i, idx_j=n_idx_j)
                    loss_far = criterion(xi=far_sub[:half], xj=far_sub[half:], sub_idx=far_sub_idx, x_i=far_match[:half],
                                         x_j=far_match[half:], label=far_sim, match_idx=far_match_idx, weight=far_weight,
                                         idx_i=f_idx_i, idx_j=f_idx_j)
                    loss = loss_near + loss_far
                    model_optim.zero_grad()
                    loss.backward()
                    model_optim.step()
                    epoch_loss += loss.item()
            if cfg.lorenz > 0:
                if epoch % 1== 0:
                    model.lorenz.both_train(False, True)
                    with tqdm(dataloader,'lorentz') as tq:
                        for near, far in tq:
                            near_traj, near_sim, near_sub_idx, near_match_idx, near_weight, n_idx_i, n_idx_j = near
                            far_traj, far_sim, far_sub_idx, far_match_idx, far_weight, f_idx_i, f_idx_j = far
                            near_sub, near_match = model(near_traj)  # vecters [B*SAM, d_model]
                            far_sub, far_match = model(far_traj)
                            half = near_sub.shape[0] // 2
                            loss_near = criterion(xi=near_sub[:half], xj=near_sub[half:], sub_idx=near_sub_idx,
                                                  x_i=near_match[:half], x_j=near_match[half:], label=near_sim,
                                                  match_idx=near_match_idx, weight=near_weight, idx_i=n_idx_i, idx_j=n_idx_j)
                            loss_far = criterion(xi=far_sub[:half], xj=far_sub[half:], sub_idx=far_sub_idx,
                                                 x_i=far_match[:half],
                                                 x_j=far_match[half:], label=far_sim, match_idx=far_match_idx,
                                                 weight=far_weight,
                                                 idx_i=f_idx_i, idx_j=f_idx_j)
                            loss = loss_near + loss_far
                            model_optim.zero_grad()
                            loss.backward()
                            model_optim.step()
                            epoch_loss += loss.item()


        emb = valid(validloader, model)
        # emb = valid(validloader, model)

        print(f"gpu alloc:{torch.cuda.max_memory_allocated() / (1024 ** 3)}|"
              f"resv:{torch.cuda.max_memory_reserved() / (1024 ** 3)}|cache:{torch.cuda.max_memory_cached() / (1024 ** 3)}")


        hr10, hr50, hr5, ndcg, hr10in50, metric = cal_top10_acc(dist, emb, (int(cfg.train_ratio*len_traj), int(cfg.valid_ratio* len_traj)), model.lorenz, _10in50=True, quick=False)
        print(f"gpu alloc:{torch.cuda.max_memory_allocated() / (1024 ** 3)}|"
              f"resv:{torch.cuda.max_memory_reserved() / (1024 ** 3)}|cache:{torch.cuda.max_memory_cached() / (1024 ** 3)}")
        if hr10 >= bst10:
            bst10 = hr10
            early_stop = 0
            torch.save(model.state_dict(), f'../data_set/{cfg.city}/model_{cfg.sim}_{cfg.lorenz}_best.pth')
            pickle.dump(metric, open(f'traj2simvec_metric_{cfg.city}_{cfg.sim}_l{cfg.lorenz}_acc{hr10}', 'wb'))
            print(f'Best HR10:{hr10:.4f} HR5:{bst5:.4f} HR50:{bst50:.4f} NDCG:{bndcg:.7f}, HR10in50:{bst10in50:.4f} save {cfg.city}/model_{cfg.sim}_{cfg.lorenz}_best.pth, quick cal sim')
        else:
            early_stop += 1
            if early_stop >= 3:
                break
            print(f'worse HR10:{hr10:.4f}, quick cal sim')
        bst5, bst50, bndcg, bst10in50 = max(bst5, hr5), max(bst50, hr50), max(bndcg, ndcg), max(bst10in50, hr10in50)
        # print(f'HR5:{bst5:.4f},HR10:{bst10:.4f}, HR50:{bst50:.4f}, NDCG:{bndcg:.7f}, HR10in50:{bst10in50:.4f}')
    print(f'Best HR10:{bst10:.4f}, HR50:{bst50:.4f}, HR5:{bst5:.4f}, NDCG:{bndcg:.7f}, HR10in50:{bst10in50:.4f}')
    model.load_state_dict(torch.load(f'../data_set/{cfg.city}/model_{cfg.sim}_{cfg.lorenz}_best.pth'))

    emb = valid(validloader, model)
    hr10, hr50, hr5, ndcg, hr10in50,_ = cal_top10_acc(dist, emb, (int(cfg.valid_ratio*len_traj), len_traj), model.lorenz, _10in50=True)

    print(f'Final HR10:{hr10:.4f}, HR50:{hr50:.4f}, HR5:{hr5:.4f}, NDCG:{ndcg:.7f}, HR10in50:{hr10in50:.4f}')

def parse_args():
    parser = argparse.ArgumentParser(description="Traj2SimVec")
    parser.add_argument("-L", "--lorenz", type=int, default=0)
    parser.add_argument("-c", "--city", type=str, default='chengdu')
    parser.add_argument("-s", "--sim", type=str, default='dtw')
    parser.add_argument("-q", "--sqrt", type=float, default=8)
    #parser.add_argument("-m", "--mod", type=str, default='lstm')
    parser.add_argument("-r", "--ratio", type=float, default=1.0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg.sim, cfg.city, cfg.lorenz, cfg.sqrt, cfg.ratio = args.sim, args.city, args.lorenz, args.sqrt, args.ratio

    train(load=False)
    # values = [5, 2, 9, 1, 5, 6, 7, 3, 2, 0, 8, 7, 6, 5, 3, 4, 9, 8, 1, 0]
    # idx = list(range(200, 220))  # Indices from 0 to 19
    #
    # # Sorting 'values' and updating 'idx' accordingly
    # sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
    # sorted_values = [values[i] for i in sorted_indices]
    # sorted_idx = [idx[i] for i in sorted_indices]
    #
    # print("Sorted values:", sorted_values)
    # print("Corresponding sorted indices:", sorted_idx)
    #
    # trajectories = [
    #     [[1, 2], [3, 4]],
    #     [[5, 6], [7, 8], [9, 10]],
    #     [[11, 12]]
    # ]
    #
    # # Finding the maximum length of the trajectories
    # max_len = max(len(traj) for traj in trajectories)
    #
    # # Padding each trajectory with [0, 0] to match the maximum length
    # padded_trajectories = [traj + [[0, 0]] * (max_len - len(traj)) for traj in trajectories]
    #
    # print(padded_trajectories)

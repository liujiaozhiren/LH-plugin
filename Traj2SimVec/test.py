import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader
from Traj2SimVec import cfg
from Traj2SimVec.network import Traj2SimVecHandler
from lorenz.transfer import cal_top10_acc

device='cuda' if torch.cuda.is_available() else 'cpu'

def pload(file_path):
    with open(file_path, "rb") as tar:
        out = pickle.load(tar)
    return out

def preproc_valid(trajs):
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

def data_enhance(trajs):
    xs, ys = [], []
    for traj in trajs:
        for item in traj:
            x, y = item
            xs.append(x)
            ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    meanx, meany, stdx, stdy = xs.mean(), ys.mean(), xs.std(), ys.std()
    for traj in trajs:
        for idx in range(len(traj)):
            x, y = traj[idx]
            traj[idx] = [(x - meanx) / stdx, (y - meany) / stdy]

def valid(dataloader, model,l):
    model.eval()
    embs = torch.zeros((l, 128)).to(device)
    start = 0
    with torch.no_grad():
        for trajs, lens in dataloader:
            y, _ = model(trajs)
            emb = y[torch.arange(len(lens)), lens]
            embs[start:start + len(emb)] = emb
            start += len(emb)
    return embs

dist = pload(f'../data_set/{cfg.city}/{cfg.sim}_test.pkl')
dist = np.array(dist)

path = '../data'
trajs = pload(f'../data_set/{cfg.city}/trajs_test.pkl')

data_enhance(trajs)
data_valid = preproc_valid(trajs)
model = Traj2SimVecHandler(cfg, trajs).to(device)
validloader = DataLoader(data_valid, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn_valid)
model.load_state_dict(torch.load(f'../data_set/{cfg.city}/model_{cfg.sim}_{cfg.lorenz}_best.pth'))
emb = valid(validloader, model, len(trajs))

acc = cal_top10_acc(dist, emb, [0,len(trajs)], lorenz=model.lorenz, config_lorenz=0, quick=True, _10in50=False)
print(acc[0])

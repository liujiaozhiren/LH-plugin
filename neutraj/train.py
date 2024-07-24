import argparse
import os
import random

import numpy as np
import torch

import tools.config as config
from geo_rnns.neutraj_trainer import NeuTrajTrainer
from neutraj.tools import preprocess


def set_seed(seed=-1):
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="neutraj")
    parser.add_argument("-L", "--lorenz", type=int, default=0)
    parser.add_argument("-c", "--city", type=str, default='chengdu')
    parser.add_argument("-s", "--sim", type=str, default='dtw')
    parser.add_argument("-q", "--sqrt", type=float, default=8)
    parser.add_argument("-m", "--mod", type=str, default='lstm')
    parser.add_argument('-d', "--dim", type=int, default=96)
    parser.add_argument('-r', "--ratio", type=float, default=1.0)
    parser.add_argument('-C', "--Const", type=float, default=1.0)

    args = parser.parse_args()
    config.lorenz = args.lorenz
    config.fname = args.city
    config.distance_type = args.sim
    config.sqrt = args.sqrt
    config.lorenz_mod = args.mod
    config.d = args.dim
    config.C = args.Const
    # config.ratio = args.ratio
    config.seeds_radio = args.ratio * config.seeds_radio
    config.update()

    set_seed(2024)
    # preprocess.trajectory_feature_generation(path=config.base_data_path)
    print('os.environ["CUDA_VISIBLE_DEVICES"]= {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print(config.config_to_str())
    coor_pkl_path, data_name = preprocess.trajectory_feature_generation(config,)
    print("finish")
    trajrnn = NeuTrajTrainer(tagset_size=config.d, batch_size=config.batch_size,
                             sampling_num=config.sampling_num)
    trajrnn.data_prepare(griddatapath=config.gridxypath, coordatapath=config.corrdatapath,
                         distancepath=config.distancepath, train_radio=config.seeds_radio)

    trajrnn.neutraj_train(load_model=None, in_cell_update=config.incell,
                          stard_LSTM=config.stard_unit)

    acc1 = trajrnn.trained_model_eval(
        load_model=f"./model/{config.data_type}_{config.distance_type}_{config.lorenz}_best_model.h5")
    print(acc1)


def train(lorenz=False):
    preprocess.trajectory_feature_generation(path=config.base_data_path)
    config.lorenz = lorenz
    # print('os.environ["CUDA_VISIBLE_DEVICES"]= {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print(config.config_to_str())
    trajrnn = NeuTrajTrainer(tagset_size=config.d, batch_size=config.batch_size,
                             sampling_num=config.sampling_num)
    trajrnn.data_prepare(griddatapath=config.gridxypath, coordatapath=config.corrdatapath,
                         distancepath=config.distancepath, train_radio=config.seeds_radio)

    trajrnn.neutraj_train(load_model=None, in_cell_update=config.incell,
                          stard_LSTM=config.stard_unit)

    acc1 = trajrnn.trained_model_eval(load_model="model/best_model.h5")
    print(acc1)

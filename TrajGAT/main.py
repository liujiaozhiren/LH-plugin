import argparse
import logging
import random

import numpy as np
import yaml
import warnings

warnings.filterwarnings("ignore")

import torch

from exp.exp_GraphTransformer import ExpGraphTransformer

from datetime import datetime, timezone, timedelta
def set_seed(seed=-1):
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(2024)
    parser = argparse.ArgumentParser(description="TrajGAT")
    parser.add_argument("-C", "--config", type=str, default='model_config.yaml')
    parser.add_argument("-G", "--gpu", type=str, default="0")
    parser.add_argument("-L1", "--load-model", type=str, default=None)
    parser.add_argument("-J", "--just_embedding", action="store_true")
    parser.add_argument("-L","--lorenz", type=float, default=0.0)
    parser.add_argument("-c","--city", type=str, default='chengdu')
    parser.add_argument("-s","--sim", type=str, default='dtw')
    parser.add_argument("-q", "--sqrt", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("-i", "--init",type=float, default=0.0)
    parser.add_argument('-r', '--ratio', type=float, default=1.0)
    args = parser.parse_args()

    lorenz = args.lorenz
    city = args.city
    sim = args.sim
    lr = args.lr
    init = args.init if args.init != 0.0 else None

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config['lorenz'] = lorenz
    config['dis_type'] = sim
    config['data'] = city
    config['init_lr'] = lr
    config['model_type'] = "lstm"
    config['sqrt'] = args.sqrt
    config['init'] = init
    config['train_data_range'][1]= int(config['train_data_range'][1] * args.ratio)
    print("Args in experiment:")
    print(config)
    # print("GPU:", args.gpu)
    # print("Load model:", args.load_model)
    # print("Store embeddings:", args.just_embedding, "\n")
    date = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S") + '.log'
    logging.basicConfig(level=logging.DEBUG,
                        format="[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
                        handlers=[logging.FileHandler(
                              f"./runs/{city}_{sim}_lorenz{lorenz}_{date}",
                            mode='w'), logging.StreamHandler()])

    if args.just_embedding:
        ExpGraphTransformer(config=config, gpu_id=args.gpu, load_model=args.load_model,
                            just_embeddings=args.just_embedding).embedding()
    else:
        ExpGraphTransformer(config=config, gpu_id=args.gpu, load_model=args.load_model,
                            just_embeddings=args.just_embedding).train()

    torch.cuda.empty_cache()

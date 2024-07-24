import argparse

from Trainer import STsim_Trainer
import torch
import os

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    args = argparse.ArgumentParser(description="ST2Vec")
    args.add_argument("-L", "--lorenz", type=int, default=0)
    args.add_argument('-c', '--config', type=str, default='config.yaml')
    args.add_argument('-r', '--ratio', type=float, default=1.0)

    args = args.parse_args()
    # train and test
    cfg = args.config
    ratio = args.ratio
    STsim = STsim_Trainer(cfg)

    load_model_name = None
    load_optimizer_name = None


    STsim.ST_train(load_model=load_model_name, load_optimizer=load_optimizer_name,lorenz=args.lorenz, ratio=ratio)

    # STsim.ST_eval(load_model=load_model_name)


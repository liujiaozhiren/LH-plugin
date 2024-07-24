import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dim=128
batch_size=10
city='chengdu'
sim='dtw'
lorenz=0
sqrt=8
ratio=1.0

train_ratio = 0.6

valid_ratio = 0.8

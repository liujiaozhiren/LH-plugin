import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dim=256
batch_size=10
city='chengdu'
sim='dtw'
lorentz=0
sqrt=8
ratio=1.0


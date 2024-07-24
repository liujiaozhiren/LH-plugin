import torch
import torch.nn as nn

import sys
sys.path.append('../')
from lorenz.transfer import Lorenz



class Traj2SimVecHandler(nn.Module):
    def __init__(self, cfg, trajs):
        super(Traj2SimVecHandler, self).__init__()
        self.traj2simvec = Traj2SimVec(cfg.dim)
        #torch.save(self.state_dict(),'model_l0.h5')
        #self.lorenz = Lorenz(base_model=[self.traj2simvec], dim=(2, 128), lorenz=cfg.lorenz, trajs=None, sqrt=cfg.sqrt)
        #torch.save(self.state_dict(),'model_l1.h5')
        self.lorenz = Lorenz(base_model=[self.traj2simvec], dim=(2, 128), lorenz=cfg.lorenz, trajs=trajs, sqrt=cfg.sqrt)
    def forward(self,x):
        return self.traj2simvec(x)


class Traj2SimVec(nn.Module):
    def __init__(self, dim):
        super(Traj2SimVec, self).__init__()
        # self.baselinear = nn.Linear(2, dim)
        self.baseRNN = nn.LSTM(2, dim, 2, batch_first=True)
        self.subPart = subPart(dim)
        self.subPart2 = subPart2(dim*2)

    def forward(self, x):
        # x = self.baselinear(x)
        x, _ = self.baseRNN(x)
        x0 = self.subPart(x)
        x1 = self.subPart2(x)
        return x0, x1


class subPart(nn.Module):
    def __init__(self, dim):
        super(subPart, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.linear3 = nn.Linear(dim, dim)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        C = nn.functional.sigmoid(x1) * nn.functional.tanh(x2)
        x3 = self.linear3(x)
        v = x + nn.functional.sigmoid(x3) * nn.functional.tanh(C)
        return v

class subPart2(nn.Module):
    def __init__(self, dim):
        super(subPart2, self).__init__()
        self.baseRNN = nn.LSTM(dim, dim, 2, batch_first=True)

    def forward(self, x):
        assert len(x.shape) == 3

        last_element = x[:, -1, :].unsqueeze(1)
        x_tmp = torch.cat([x, last_element.repeat(1, x.shape[1], 1)],dim=2)
        v, _ = self.baseRNN(x_tmp)
        return v

if __name__ == '__main__':
    N, L, H = 2, 3, 4  # Re-defining dimensions for clarity
    tensor = torch.randn(N, L, 2)  # Re-generating the example tensor

    model = Traj2SimVec(H)
    output = model(tensor)
    print(output[0].shape)
    print(output[1].shape)



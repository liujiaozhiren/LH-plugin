import threading

import torch
from torch import nn

from lorentz.transfer import Lorentz


class LH_trainer:
    def __init__(self, model, lorentz, every_epoch=1, loss_cmb=1, grad_reduce=0.1):
        self.lorentz = lorentz
        self.every_epoch = every_epoch
        self.loss_cmb = loss_cmb
        self.grad_reduce = grad_reduce
        self.iteration = 0
        self.model = model



    def get_iter(self, epoch):
        return LH_iter(epoch, self.model, self.lorentz, self.every_epoch, self.grad_reduce, self.loss_cmb)


class LH_iter:
    def __init__(self, epoch, model, lorentz, every_epoch, grad_reduce, loss_cmb):
        self.epoch = epoch
        self.model = model
        self.every_epoch = every_epoch
        #self.loss_cmb = loss_cmb
        self.lorentz = lorentz
        self.grad_reduce = grad_reduce
        self.loss_cmb = loss_cmb
        self.iteration = 0

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def __next__(self):
        if self.iteration >= 2:
            raise StopIteration
        self.iteration += 1
        if self.iteration == 1:
            self.model.lorentz.both_train(True, False)
            return 'normal_train', 1
        elif self.iteration == 2:
            if self.lorentz == 0:
                raise StopIteration
            if self.epoch % self.every_epoch == self.every_epoch - 1:
                self.model.lorentz.both_train(False, True)
            return 'lh_train', self.loss_cmb

    def LH_grad_reduce(self):
        if self.iteration == 2:
            for param in self.model.lorentz.traj2vec.parameters():
                if param.grad is not None:
                    param.grad.data *= 0.1




class EmbeddingModelHandler(nn.Module):
    def __init__(self, GivenModel, lh_input_size,lh_target_size, lh_lorentz, lh_trajs=None, lh_model_type='lstm', lh_sqrt=8.0, lh_C=1):
        super().__init__()
        self._init = False
        # exclude the kwargs keys start with 'lh_'
        # model_kwargs = {k: v for k, v in kwargs.items() if not k.startswith('lh_')}
        self.model = GivenModel
        lorentz = Lorentz(base_model=[self.model], dim=(lh_input_size, lh_target_size), lorentz=lh_lorentz, trajs=lh_trajs, load=None, model_type=lh_model_type, sqrt=lh_sqrt, C=lh_C)
        self.lorentz = lorentz
        self.__dict__["lorentz"] = lorentz
        self.__dict__["model"] = GivenModel
        self._init = True


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if name == 'model' or name == 'lorentz':
                if not self._init:
                    raise AttributeError(f"'{type(self).__name__}' object has no init '{name}'")
            try:
                return getattr(self.model, name)
            except AttributeError:
                # 如果 B 中没有这个属性，正常抛出 AttributeError
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def traj_reload(self, network, traj_seqs, time_seqs): ## using: if traj embedding need update
        s_input, seq_lengths = self.model.stEncoder.embedding_S(network, traj_seqs)
        t_input = self.model.stEncoder.embedding_T(time_seqs)
        self.lorentz.dynamic_store_traj_data(torch.cat([s_input, t_input], dim=-1).detach(), seq_lengths)


class Handler:
    def __init__(self, model):
        self.model = model
        self.a = None

    def __getattr__(self, name):
        # 检查属性是否真的存在于model中
        if hasattr(self.model, name):
            return getattr(self.model, name)
        else:
            # 如果model没有这个属性，直接抛出AttributeError
            raise AttributeError(f"{type(self.model).__name__} object has no attribute {name}")

# 假设的Model类
class Model:
    def __init__(self):
        self.foo = 'bar'
if __name__ == '__main__':
    model = Model()
    handler = Handler(model)

    print(handler.foo)  # 正常输出
    print(handler.a)  # 应该抛出AttributeError

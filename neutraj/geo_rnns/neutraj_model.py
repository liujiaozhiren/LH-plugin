import os
import sys

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn import Module

import tools.config as config
from geo_rnns.sam_cells import SAM_LSTMCell, SAM_GRUCell

parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_parent_dir)

from lorentz.transfer import Lorentz


class RNNEncoder(Module):
    def __init__(self, input_size, hidden_size, grid_size, stard_LSTM=False, incell=True):
        super(RNNEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.stard_LSTM = stard_LSTM
        if config.device == 'cuda':
            if self.stard_LSTM:
                if config.recurrent_unit == 'GRU':
                    self.cell = torch.nn.GRUCell(input_size - 2, hidden_size).cuda()
                elif config.recurrent_unit == 'SimpleRNN':
                    self.cell = torch.nn.RNNCell(input_size - 2, hidden_size).cuda()
                else:
                    self.cell = torch.nn.LSTMCell(input_size - 2, hidden_size).cuda()
            else:
                if config.recurrent_unit == 'GRU':
                    self.cell = SAM_GRUCell(input_size, hidden_size, grid_size, incell=incell).cuda()
                elif config.recurrent_unit == 'SimpleRNN':
                    self.cell = SpatialRNNCell(input_size, hidden_size, grid_size, incell=incell).cuda()
                else:
                    self.cell = SAM_LSTMCell(input_size, hidden_size, grid_size, incell=incell).cuda()
        else:
            if self.stard_LSTM:
                if config.recurrent_unit == 'GRU':
                    self.cell = torch.nn.GRUCell(input_size - 2, hidden_size).cpu()
                elif config.recurrent_unit == 'SimpleRNN':
                    self.cell = torch.nn.RNNCell(input_size - 2, hidden_size).cpu()
                else:
                    self.cell = torch.nn.LSTMCell(input_size - 2, hidden_size).cpu()
            else:
                if config.recurrent_unit == 'GRU':
                    self.cell = SAM_GRUCell(input_size, hidden_size, grid_size, incell=incell).cpu()
                elif config.recurrent_unit == 'SimpleRNN':
                    self.cell = SpatialRNNCell(input_size, hidden_size, grid_size, incell=incell).cpu()
                else:
                    self.cell = SAM_LSTMCell(input_size, hidden_size, grid_size, incell=incell).cpu()

        #print(self.cell)
        #print('in cell update: {}'.format(incell))
        # self.cell = torch.nn.LSTMCell(input_size-2, hidden_size).cuda()

    def forward(self, inputs_a, initial_state=None):
        inputs, inputs_len = inputs_a
        time_steps = inputs.size(1)
        out = None
        if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
            out = initial_state
        else:
            out, state = initial_state

        outputs = []
        for t in range(time_steps):
            if self.stard_LSTM:
                cell_input = inputs[:, t, :][:, :-2]
            else:
                cell_input = inputs[:, t, :]
            if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
                out = self.cell(cell_input, out)
            else:
                out, state = self.cell(cell_input, (out, state))
            outputs.append(out)
        mask_out = []
        for b, v in enumerate(inputs_len):
            mask_out.append(outputs[v - 1][b, :].view(1, -1))
        return torch.cat(mask_out, dim=0)

    def batch_grid_state_gates(self, inputs_a, initial_state=None):
        inputs, inputs_len = inputs_a
        time_steps = inputs.size(1)
        out, state = initial_state
        outputs = []
        gates_out_all = []
        if config.device == 'cuda':
            batch_weight_ih = autograd.Variable(self.cell.weight_ih.data, requires_grad=False).cuda()
            batch_weight_hh = autograd.Variable(self.cell.weight_hh.data, requires_grad=False).cuda()
            batch_bias_ih = autograd.Variable(self.cell.bias_ih.data, requires_grad=False).cuda()
            batch_bias_hh = autograd.Variable(self.cell.bias_hh.data, requires_grad=False).cuda()
        else:
            batch_weight_ih = autograd.Variable(self.cell.weight_ih.data, requires_grad=False).cpu()
            batch_weight_hh = autograd.Variable(self.cell.weight_hh.data, requires_grad=False).cpu()
            batch_bias_ih = autograd.Variable(self.cell.bias_ih.data, requires_grad=False).cpu()
            batch_bias_hh = autograd.Variable(self.cell.bias_hh.data, requires_grad=False).cpu()
        for t in range(time_steps):
            # cell_input = inputs[:, t, :][:,:-2]
            cell_input = inputs[:, t, :]
            self.cell.update_memory(cell_input, (out, state),
                                    batch_weight_ih, batch_weight_hh,
                                    batch_bias_ih, batch_bias_hh)


class NeuTraj_Network(Module):
    def __init__(self, input_size, target_size, grid_size, batch_size, sampling_num, stard_LSTM=False, incell=True):
        super(NeuTraj_Network, self).__init__()
        self.input_size = input_size
        self.target_size = target_size
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.sampling_num = sampling_num

        if config.device == 'cuda':
            if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
                self.hidden = autograd.Variable(
                    torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                    requires_grad=False).cuda()
            else:
                self.hidden = (
                autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                                  requires_grad=False).cuda(),
                autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                                  requires_grad=False).cuda())
            self.rnn = RNNEncoder(self.input_size, self.target_size, self.grid_size, stard_LSTM=stard_LSTM,
                                  incell=incell).cuda()
        else:
            if config.recurrent_unit == 'GRU' or config.recurrent_unit == 'SimpleRNN':
                self.hidden = autograd.Variable(
                    torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                    requires_grad=False).cpu()
            else:
                self.hidden = (
                    autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                                      requires_grad=False).cpu(),
                    autograd.Variable(torch.zeros(self.batch_size * (1 + self.sampling_num), self.target_size),
                                      requires_grad=False).cpu())
            self.rnn = RNNEncoder(self.input_size, self.target_size, self.grid_size, stard_LSTM=stard_LSTM,
                                  incell=incell).cpu()
        # self.lorentz = Lorentz(base_model=[self.rnn], dim=(2, target_size), lorentz=config.lorentz, trajs=trajs, load=None, model_type=config.lorentz_mod, sqrt=config.sqrt, C=config.C)


    def forward(self, inputs_arrays, inputs_len_arrays):
        anchor_input = torch.Tensor(inputs_arrays[0])
        trajs_input = torch.Tensor(inputs_arrays[1])
        negative_input = torch.Tensor(inputs_arrays[2])

        anchors_idx, trajs_idx, negs_idx = inputs_arrays[-1]

        anchor_input_len = inputs_len_arrays[0]
        trajs_input_len = inputs_len_arrays[1]
        negative_input_len = inputs_len_arrays[2]

        if config.device == 'cuda':
            anchor_embedding = self.rnn([autograd.Variable(anchor_input, requires_grad=False).cuda(), anchor_input_len],
                                        self.hidden)
            trajs_embedding = self.rnn([autograd.Variable(trajs_input, requires_grad=False).cuda(), trajs_input_len],
                                       self.hidden)
            negative_embedding = self.rnn(
                [autograd.Variable(negative_input, requires_grad=False).cuda(), negative_input_len], self.hidden)
        else:
            anchor_embedding = self.rnn([autograd.Variable(anchor_input, requires_grad=False).cpu(), anchor_input_len],
                                        self.hidden)
            trajs_embedding = self.rnn([autograd.Variable(trajs_input, requires_grad=False).cpu(), trajs_input_len],
                                       self.hidden)
            negative_embedding = self.rnn(
                [autograd.Variable(negative_input, requires_grad=False).cpu(), negative_input_len], self.hidden)
        # trajs_loss = torch.exp(-F.pairwise_distance(anchor_embedding, trajs_embedding, p=2))
        # negative_loss = torch.exp(-F.pairwise_distance(anchor_embedding, negative_embedding, p=2))

        # trajs_loss = torch.exp(-self.lorentz.dist(anchor_embedding, trajs_embedding, config.lorentz))
        # negative_loss = torch.exp(-self.lorentz.dist(anchor_embedding, negative_embedding, config.lorentz))
        #
        return anchor_embedding, trajs_embedding, negative_embedding

        if config.lorentz == 0:
            trajs_loss = torch.exp(-F.pairwise_distance(anchor_embedding, trajs_embedding, p=2))
            negative_loss = torch.exp(-F.pairwise_distance(anchor_embedding, negative_embedding, p=2))
        else:
            trajs_loss = torch.exp(-self.lorentz.learned_cmb_dist(anchor_embedding, trajs_embedding, anchors_idx, trajs_idx))
            negative_loss = torch.exp(-self.lorentz.learned_cmb_dist(anchor_embedding, negative_embedding, anchors_idx, negs_idx))
        assert anchor_embedding.isnan().any() == False and anchor_embedding.isinf().any() == False
        assert trajs_embedding.isnan().any() == False and trajs_embedding.isinf().any() == False
        assert trajs_loss.isinf().any() == False
        assert negative_loss.isinf().any() == False

        return trajs_loss, negative_loss

    def spatial_memory_update(self, inputs_arrays, inputs_len_arrays):
        batch_traj_input = torch.Tensor(inputs_arrays[3])
        batch_traj_len = inputs_len_arrays[3]
        if config.device == 'cuda':
            batch_hidden = (
                autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size), requires_grad=False).cuda(),
                autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size), requires_grad=False).cuda())
            self.rnn.batch_grid_state_gates([autograd.Variable(batch_traj_input).cuda(), batch_traj_len], batch_hidden)
        else:
            batch_hidden = (
                autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size), requires_grad=False).cpu(),
                autograd.Variable(torch.zeros(len(batch_traj_len), self.target_size), requires_grad=False).cpu())
            self.rnn.batch_grid_state_gates([autograd.Variable(batch_traj_input).cpu(), batch_traj_len], batch_hidden)

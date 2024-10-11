# Data path
import torch.cuda

fname = 'chengdu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
distance_type = 'dtw'
base_data_path = f'../data_set/{fname}/trajs_10000.pkl'
neutraj_data_tmp = '../data_set/neutraj/'

corrdatapath = neutraj_data_tmp + f'{fname}_traj_coord'
gridxypath = neutraj_data_tmp + f'{fname}_traj_grid'

distancepath = f'../data_set/{fname}/{distance_type}_10000x10000.pkl'

log_root = f'./log/'

traj_index_path = neutraj_data_tmp + f'{fname}_traj_index'
# Training Prarmeters
GPU = "0"
learning_rate = 0.01
seeds_radio = 0.6
epochs = 100
batch_size = 25
sampling_num = 10
lorentz_mod = 'lstm'

# distance_type = distancepath.split('/')[2].split('_')[1]
# data_type = distancepath.split('/')[2].split('_')[0]
data_type = fname
if distance_type == 'dtw':
    mail_pre_degree = 16
else:
    mail_pre_degree = 8

# Test Config
datalength = 10000
em_batch = 500
test_num = 100
sqrt=8
C = 1
# Model Parameters
d = 96
stard_unit = False  # It controls the type of recurrent unit (standrad cells or SAM argumented cells)
incell = True
recurrent_unit = 'GRU'  # GRU, LSTM or SimpleRNN
spatial_width = 2
ratio=1.0

gird_size = [500, 500]
lorentz = True

picked=0
picked_idxs = None
def config_to_str():
    configs = 'learning_rate = {} '.format(learning_rate) + '\n' + \
              'mail_pre_degree = {} '.format(mail_pre_degree) + '\n' + \
              'seeds_radio = {} '.format(seeds_radio) + '\n' + \
              'epochs = {} '.format(epochs) + '\n' + \
              'datapath = {} '.format(corrdatapath) + '\n' + \
              'datatype = {} '.format(data_type) + '\n' + \
              'corrdatapath = {} '.format(corrdatapath) + '\n' + \
              'distancepath = {} '.format(distancepath) + '\n' + \
              'distance_type = {}'.format(distance_type) + '\n' + \
              'recurrent_unit = {}'.format(recurrent_unit) + '\n' + \
              'batch_size = {} '.format(batch_size) + '\n' + \
              'sampling_num = {} '.format(sampling_num) + '\n' + \
              'incell = {}'.format(incell) + '\n' + \
              'lorentz = {}'.format(lorentz) + '\n' + \
              'dim = {}'.format(d) + '\n' + \
              'gird_size = {}'.format(gird_size) + '\n' + \
              'stard_unit = {}'.format(stard_unit) + '\n' + \
              'sqrt = {}'.format(sqrt) + '\n' + \
              'ratio = {}'.format(ratio) + '\n' + \
              'C = {}'.format(C)

    return configs


def update():
    global base_data_path, corrdatapath, gridxypath, distancepath, log_root, traj_index_path, data_type, mail_pre_degree
    base_data_path = f'../data_set/{fname}/trajs_10000.pkl'
    corrdatapath = neutraj_data_tmp + f'{fname}_traj_coord'
    gridxypath = neutraj_data_tmp + f'{fname}_traj_grid'
    distancepath = f'../data_set/{fname}/{distance_type}_10000x10000.pkl'
    traj_index_path = neutraj_data_tmp + f'{fname}_traj_index'

    data_type = fname
    if distance_type == 'dtw':
        mail_pre_degree = 16
    else:
        mail_pre_degree = 8


if __name__ == '__main__':
    print('../model/model_training_600_{}_acc_{}'.format((0), 1))
    print(config_to_str())

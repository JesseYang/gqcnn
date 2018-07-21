from easydict import EasyDict as edict
import numpy as np

cfg = edict()

# datapoints
cfg.filename = 'dexnet2.hdf5'
cfg.data_dir = '3dnet_kit_06_13_17'
cfg.train_pct = 0.8
cfg.tot_datapoints = 6728 * 1000 + 850
cfg.train_num = int(cfg.tot_datapoints * cfg.train_pct)
cfg.val_num = cfg.tot_datapoints - cfg.train_num

cfg.im_height = 32
cfg.im_width = 32

# weight_decay
cfg.weight_decay = 0.0005

# lr_decay
cfg.base_lr = 0.01
cfg.decay_step_multiplier = 0.2
cfg.decay_step = cfg.decay_step_multiplier * cfg.train_num
cfg.decay_rate = 0.95
# optimizer
cfg.momentum_rate = 0.9

# train
cfg.epoch_num = 25

# LocalNorm params
cfg.radius = 2
cfg.alpha = 2e-5
cfg.beta = 0.75
cfg.bias = 1.0

# model architecture
cfg.drop_fc3 = False
cfg.fc3_drop_rate = 0

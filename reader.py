import collections
from scipy import misc
import tensorflow as tf

import os, sys, shutil
import numpy as np
import random
import copy
import logging
import cv2
import uuid
try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg

from tensorpack import *

import h5py

SAVE_DIR = 'input_images'

class Data(RNGDataFlow):
    def __init__(self,
                 filename,
                 save_data=False,
                 test_set=False):

        self.test_set = test_set
        self.save_data = save_data

        self.filename = filename

    def size(self):
        return cfg.train_num if not self.test_set else cfg.val_num

    def generate_sample(self, idx):
        with h5py.File(self.filename, 'r') as f:
            depth_im = f['depth_im'][idx]
            depth = f['hand_depth'][idx]
            label = f['label'][idx]            

        if self.save_data:
            misc.imsave(os.path.join(SAVE_DIR, '%d_%d_depth_im.jpg' % (label, idx)), depth_im[:,:,0])

        return [depth_im, depth, label]

    def get_data(self):
        if not self.test_set:
            idxs = np.arange(cfg.train_num)
            self.rng.shuffle(idxs)
        else:
            idxs = np.arange(cfg.train_num, cfg.tot_datapoints)
        for k in idxs:
            retval = self.generate_sample(k)
            if retval == None:
                continue
            yield retval

    def reset_state(self):
        super(Data, self).reset_state() 

if __name__ == '__main__':

    ds = Data(cfg.filename, save_data=True)       
    ds.reset_state()

    g = ds.get_data()
    for i in range(10):
        data = next(g)
        import pdb
        pdb.set_trace()

import pdb
import cv2
import sys
import argparse
import numpy as np
import os
import shutil
import multiprocessing
from abc import abstractmethod

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.ops import init_ops
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.common import get_tf_version_number
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.models import (
    Conv2D, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected,
    LinearWrap)

from cfgs.config import cfg
from reader import Data

# @layer_register()
# def LocalNorm(name, input,  depth_radius=5, bias=1, alpha=1, beta=0.5):
def LocalNorm(input):
    input = tf.nn.relu(input)
    return tf.nn.local_response_normalization(input, cfg.radius, cfg.bias, cfg.alpha, cfg.beta)

class GQCNN(ModelDesc):
    def __init__(self, data_format='NHWC'):
        super(GQCNN, self).__init__()
        self.data_format = data_format

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, cfg.im_height, cfg.im_width, 1], 'input'), 
                InputDesc(tf.float32, [None], 'pose'),
                InputDesc(tf.int32, [None], 'label')]

    def _get_logits(self, image, pose):
        with argscope([Conv2D, MaxPooling], data_format=self.data_format, padding='same'), argscope([Conv2D, FullyConnected], activation=tf.nn.relu), argscope([Conv2D], strides=1):
            im_fc3 = (LinearWrap(image)
                       # 1_1
                       .Conv2D('conv1_1', 64, 7)
                       .MaxPooling('pool1_1', 1)
                       # 1_2
                       .Conv2D('conv1_2', 64, 5, activation=LocalNorm)
                       # .LocalNorm('conv1_2_norm', cfg.radius, cfg.alpha, cfg.beta, cfg.bias)
                       .MaxPooling('pool1_2', 2)
                       # 2_1
                       .Conv2D('conv2_1', 64, 3)
                       .MaxPooling('pool2_1', 1)
                       # 2_2
                       .Conv2D('conv2_2', 64, 3, activation=LocalNorm)
                       # .LocalNorm('conv2_2_norm', cfg.radius, cfg.alpha, cfg.beta, cfg.bias)
                       .MaxPooling('pool2_2', 1)
                       .FullyConnected('fc3', 1024)())      
            if cfg.drop_fc3:
                im_fc3 = tf.nn.dropout(fc3, cfg.fc3_drop_rate)     
            pc1 = FullyConnected('pc1', pose, 16)

        fc4_im = FullyConnected('fc4_im', im_fc3, 1024, activation=tf.identity)
        fc4_pose = FullyConnected('fc4_pose', pc1, 1024, activation=tf.identity)
        fc4 = tf.nn.relu(fc4_im + fc4_pose)
        fc5 = FullyConnected('fc5', fc4, 2)

        return fc5            

    def _build_graph(self, inputs):
        image, pose, label = inputs
        pose = tf.reshape(pose, [-1, 1])
        # image_summary = tf.cast(image, tf.uint8)
        tf.summary.image('image', image, max_outputs=3)

        logits = self._get_logits(image, pose)
        preds = tf.nn.softmax(logits)
        accuracy = tf.to_float(tf.nn.in_top_k(preds, label, 1))
        add_moving_summary(tf.reduce_mean(accuracy, name='accuracy'))

        # loss
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=label,
                                logits=logits), name='loss')
        # regularization
        wd_cost = regularize_cost('.*/W', l2_regularizer(cfg.weight_decay), name='l2_regularize_loss')
        self.cost = tf.add_n([loss, wd_cost], name='cost')

        add_moving_summary(loss, wd_cost, self.cost)

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', cfg.base_lr, summary=True)
        optimizer = tf.train.MomentumOptimizer(lr, cfg.momentum_rate, use_nesterov=True)
        return optimizer

def get_data(train_or_test, batch_size):
    is_train = train_or_test == 'train'
    ds = Data(cfg.filename, test_set=not is_train)

    augmentors = []
    ds = BatchData(ds, batch_size, remainder=not is_train)
    if is_train:
        ds = PrefetchDataZMQ(ds, min(6, multiprocessing.cpu_count()))
    return ds

def get_config(args, model):
    ds_train = get_data('train', args.batch_size_per_gpu)
    ds_test = get_data('test', args.batch_size_per_gpu)

    steps_per_epoch = cfg.train_num // args.batch_size_per_gpu

    callbacks = [
        ModelSaver(),
        PeriodicTrigger(InferenceRunner(ds_test, ScalarStats(['cost', 'accuracy'])),
                        every_k_epochs=1),
        HyperParamSetterWithFunc('learning_rate',
                               lambda e, x: cfg.base_lr * cfg.decay_rate ** (e / cfg.epoch_num) ),
        HumanHyperParamSetter('learning_rate'),
    ]

    return TrainConfig(
        dataflow=ds_train,
        callbacks=callbacks,
        model=model,
        steps_per_epoch=steps_per_epoch,
        max_epoch=cfg.epoch_num)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default="1")
    parser.add_argument('--batch_size_per_gpu', help='batch size per gpu', type=int, default=8)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--flops', action='store_true', help='print flops and exit')
    parser.add_argument('--logdir', help='train log directory name')
    args = parser.parse_args()


    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = GQCNN()
    
    if args.flops:
        input_desc = [
            InputDesc(tf.float32, [None, cfg.im_height, cfg.im_width, 1], 'input'),
            InputDesc(tf.float32, [None, 1], 'pose'),
            InputDesc(tf.int32, [None], 'label')
        ]
        input = PlaceholderInput()
        input.setup(input_desc)
        with TowerContext('', is_training=True):
            model.build_graph(*input.get_input_tensors())

        tf.profiler.profile(
            tf.get_default_graph(),
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.float_operation())
    else:
        if args.logdir != None:
            logger.set_logger_dir(os.path.join("train_log", args.logdir))
        else:
            logger.auto_set_dir()
        nr_tower = get_nr_gpu()
        config = get_config(args, model)

        if args.load:
            if args.load.endswith('npz'):
                config.session_init = DictRestore(dict(np.load(args.load)))
            else:
                config.session_init = SaverRestore(args.load)

        trainer = SyncMultiGPUTrainerParameterServer(nr_tower)
        launch_train_with_config(config, trainer)

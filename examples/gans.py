#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: gans.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import sys
import platform
import argparse
import numpy as np
import tensorflow as tf

sys.path.append('../')

from src.nets.dcgan import DCGAN
from src.nets.lsgan import LSGAN
from src.helper.trainer import Trainer
from src.helper.generator import Generator
import loader as loader
# from src.helper.visualizer import Visualizer
# import src.models.distribution as distribution

if platform.node() == 'Qians-MacBook-Pro.local':
    SAVE_PATH = '/Users/gq/tmp/'
elif platform.node() == 'arostitan':
    SAVE_PATH = '/home/qge2/workspace/data/out/gans/'
else:
    SAVE_PATH = None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--generate', action='store_true',
                        help='generate')

    parser.add_argument('--gan_type', type=str, default='dcgan',
                        help='type of GAN for experiment')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset')


    parser.add_argument('--zlen', type=int, default=100,
                        help='length of random vector z')

    parser.add_argument('--load', type=int, default=99,
                        help='Load step of pre-trained')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Init learning rate')
    parser.add_argument('--keep_prob', type=float, default=2e-4,
                        help='keep_prob')
    parser.add_argument('--bsize', type=int, default=128,
                        help='Init learning rate')
    parser.add_argument('--maxepoch', type=int, default=50,
                        help='Max iteration')

    parser.add_argument('--ng', type=int, default=1,
                        help='number generator training each step')
    parser.add_argument('--nd', type=int, default=1,
                        help='number discriminator training each step')

    return parser.parse_args()

# def preprocess_im(im):
#     """ normalize input image to [-1., 1.] """
#     im = im / 255. * 2. - 1.
#     return im

# def read_mnist_data(batch_size, n_use_label=None, n_use_sample=None):
#     """ Function for load training data 

#     If n_use_label or n_use_sample is not None, samples will be
#     randomly picked to have a balanced number of examples

#     Args:
#         batch_size (int): batch size
#         n_use_label (int): how many labels are used for training
#         n_use_sample (int): how many samples are used for training

#     Retuns:
#         MNISTData

#     """
#     data = MNISTData('train',
#                      data_dir=DATA_PATH,
#                      shuffle=True,
#                      pf=preprocess_im,
#                      # n_use_label=n_use_label,
#                      # n_use_sample=n_use_sample,
#                      batch_dict_name=['im', 'label'])
#     data.setup(epoch_val=0, batch_size=batch_size)
#     return data

# # def read_valid_data(batch_size):
# #     """ Function for load validation data """
# #     data = MNISTData('test',
# #                      data_dir=DATA_PATH,
# #                      shuffle=True,
# #                      pf=preprocess_im,
# #                      batch_dict_name=['im', 'label'])
# #     data.setup(epoch_val=0, batch_size=batch_size)
# #     return data

def train():
    FLAGS = get_args()
    if FLAGS.gan_type == 'lsgan':
        gan_model = LSGAN
        print('**** LSGAN ****')
    else:
        gan_model = DCGAN
        print('**** DCGAN ****')

    # load dataset
    if FLAGS.dataset == 'celeba':
        train_data = loader.load_celeba(FLAGS.bsize)
        im_size = 64
        n_channels = 3
    else:
        train_data = loader.load_mnist(FLAGS.bsize)
        im_size = 28
        n_channels = 1

    train_model = gan_model(input_len=FLAGS.zlen, im_size=im_size, n_channels=n_channels)
    train_model.create_train_model()

    generate_model = gan_model(input_len=FLAGS.zlen, im_size=im_size, n_channels=n_channels)
    generate_model.create_generate_model()

    trainer = Trainer(train_model, train_data,
                      moniter_gradient=False,
                      init_lr=FLAGS.lr, save_path=SAVE_PATH)
    generator = Generator(generate_model, save_path=SAVE_PATH)

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        writer = tf.summary.FileWriter(SAVE_PATH)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for epoch_id in range(FLAGS.maxepoch):
            trainer.train_epoch(sess, keep_prob=FLAGS.keep_prob,
                                n_g_train=FLAGS.ng, n_d_train=FLAGS.nd,
                                summary_writer=writer)
            generator.random_sampling(sess, plot_size=10, file_id=epoch_id)
            generator.viz_interpolate(sess, file_id=epoch_id)
            if FLAGS.zlen == 2:
                generator.viz_2D_manifold(sess, plot_size=20, file_id=epoch_id)
            saver.save(
                sess, '{}gan-{}-epoch-{}'.format(SAVE_PATH, FLAGS.gan_type, epoch_id))

        saver.save(
                sess, '{}gan-{}-epoch-{}'.format(SAVE_PATH, FLAGS.gan_type, epoch_id))


if __name__ == "__main__":
    FLAGS = get_args()

    if FLAGS.train:
        train()


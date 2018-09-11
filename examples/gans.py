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
from src.nets.began import BEGAN
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
                        help='Train the model.')
    parser.add_argument('--generate', action='store_true',
                        help='Sampling from trained model')
    parser.add_argument('--test', action='store_true',
                        help='test')

    parser.add_argument('--gan_type', type=str, default='dcgan',
                        help='Type of GAN for experiment.')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Dataset used for experiment.')

    parser.add_argument('--zlen', type=int, default=100,
                        help='length of random vector z')

    parser.add_argument('--load', type=int, default=99,
                        help='Load step of pre-trained')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Init learning rate')
    parser.add_argument('--keep_prob', type=float, default=1.,
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

def train():
    if FLAGS.gan_type == 'lsgan' or FLAGS.gan_type == 'dcgan':
        train_type_1()
    elif FLAGS.gan_type == 'began':
        train_type_2()
    else:
        raise ValueError('Wrong GAN type!')

def train_type_1():
    FLAGS = get_args()
    if FLAGS.gan_type == 'lsgan':
        gan_model = LSGAN
        print('**** LSGAN ****')
    elif FLAGS.gan_type == 'dcgan':
        gan_model = DCGAN
        print('**** DCGAN ****')
    else:
        raise ValueError('Wrong GAN type!')

    save_path = os.path.join(SAVE_PATH, FLAGS.gan_type)
    save_path += '/'

    # load dataset
    if FLAGS.dataset == 'celeba':
        train_data = loader.load_celeba(FLAGS.bsize)
        im_size = 64
        n_channels = 3
    else:
        train_data = loader.load_mnist(FLAGS.bsize)
        im_size = 28
        n_channels = 1

    train_model = gan_model(input_len=FLAGS.zlen,
                            im_size=im_size,
                            n_channels=n_channels)
    train_model.create_train_model()

    generate_model = gan_model(input_len=FLAGS.zlen,
                               im_size=im_size,
                               n_channels=n_channels)
    generate_model.create_generate_model()

    trainer = Trainer(train_model, train_data,
                      moniter_gradient=False,
                      init_lr=FLAGS.lr, save_path=save_path)
    generator = Generator(generate_model, keep_prob=FLAGS.keep_prob,
                          save_path=save_path)

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        writer = tf.summary.FileWriter(save_path)
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

            saver.save(sess, '{}gan-{}-epoch-{}'.format(save_path, FLAGS.gan_type, epoch_id))
        saver.save(sess, '{}gan-{}-epoch-{}'.format(save_path, FLAGS.gan_type, epoch_id))

def train_type_2():
    FLAGS = get_args()
    if FLAGS.gan_type == 'began':
        gan_model = BEGAN
        print('**** BEGAN ****')
    else:
        raise ValueError('Wrong GAN type!')

    save_path = os.path.join(SAVE_PATH, FLAGS.gan_type)
    save_path += '/'

    im_size = 64
    n_channels = 3
    # train_data = loader.load_mnist(FLAGS.bsize, rescale_size=None)
    train_data = loader.load_celeba(FLAGS.bsize)
    
    train_model = BEGAN(input_len=64, im_size=im_size, n_channels=n_channels)
    train_model.create_train_model()

    generate_model = BEGAN(input_len=64, im_size=im_size, n_channels=n_channels)
    generate_model.create_generate_model()

    generator = Generator(generate_model,
                          keep_prob=FLAGS.keep_prob,
                          save_path=save_path)

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        writer = tf.summary.FileWriter(save_path)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        for epoch_id in range(FLAGS.maxepoch):
            train_model.train_epoch(
                sess, train_data, init_lr=FLAGS.lr,
                n_g_train=1, n_d_train=1, keep_prob=1.0,
                summary_writer=writer)
            generator.random_sampling(sess, plot_size=10, file_id=epoch_id)
            generator.viz_interpolate(sess, file_id=epoch_id)
            saver.save(sess,'{}gan-{}-epoch-{}'.format(save_path, FLAGS.gan_type, epoch_id))
        saver.save(sess, '{}gan-{}-epoch-{}'.format(save_path, FLAGS.gan_type, epoch_id))

if __name__ == "__main__":
    FLAGS = get_args()

    if FLAGS.train:
        train()



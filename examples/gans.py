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
from src.nets.infogan import infoGAN
from src.helper.trainer import Trainer
from src.helper.generator import Generator
import loader as loader
# from src.helper.visualizer import Visualizer
# import src.models.distribution as distribution

SAVE_PATH = '/home/qge2/workspace/data/out/gans/'

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

    parser.add_argument('--w_mutual', type=float, default=1.0,
                        help='')

    return parser.parse_args()

def test():

    save_path = os.path.join(SAVE_PATH, 'test')
    save_path += '/'

    # load dataset
    if FLAGS.dataset == 'celeba':
        im_size = 32
        n_channels = 3
        n_continuous = 0
        n_discrete = 10
        cat_n_class_list = [10 for i in range(n_discrete)]

        train_data = loader.load_celeba(FLAGS.bsize, rescale_size=im_size)
    else:
        im_size = 28
        n_channels = 1
        n_continuous = 4
        n_discrete = 1
        cat_n_class_list = [10]

        train_data = loader.load_mnist(FLAGS.bsize)
        
    train_model = infoGAN(
        input_len=FLAGS.zlen, im_size=im_size, n_channels=n_channels,
        cat_n_class_list=cat_n_class_list,
        n_continuous=n_continuous, n_discrete=n_discrete,
        mutual_info_weight=FLAGS.w_mutual)
    train_model.create_train_model()

    generate_model = infoGAN(
        input_len=FLAGS.zlen, im_size=im_size, n_channels=n_channels,
        cat_n_class_list=cat_n_class_list,
        n_continuous=n_continuous, n_discrete=n_discrete)
    generate_model.create_generate_model()

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
                n_g_train=FLAGS.ng, n_d_train=FLAGS.nd, keep_prob=1.0,
                summary_writer=writer)
            for vary_discrete_id in range(n_discrete):
                generate_model.vary_discrete_sampling(
                    sess, vary_discrete_id=vary_discrete_id, plot_size=10,
                    file_id=epoch_id, save_path=save_path)
            for cont_code_id in range(n_continuous):
                generate_model.interp_cont_sampling(
                    sess, n_interpolation=10, cont_code_id=cont_code_id,
                    vary_discrete_id=0,
                    keep_prob=1., save_path=save_path, file_id=epoch_id)

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

    # init training model
    train_model = gan_model(input_len=FLAGS.zlen,
                            im_size=im_size,
                            n_channels=n_channels)
    train_model.create_train_model()

    # init generate model
    generate_model = gan_model(input_len=FLAGS.zlen,
                               im_size=im_size,
                               n_channels=n_channels)
    generate_model.create_generate_model()

    # create trainer
    trainer = Trainer(train_model, train_data,
                      moniter_gradient=False,
                      init_lr=FLAGS.lr, save_path=save_path)
    # create generator for sampling
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

    # load dataset
    if FLAGS.dataset == 'celeba':
        im_size = 128
        n_channels = 3
        train_data = loader.load_celeba(FLAGS.bsize, rescale_size=im_size)
        
    else:
        train_data = loader.load_mnist(FLAGS.bsize)
        im_size = 28
        n_channels = 1
    
    # init training model
    train_model = BEGAN(input_len=64, im_size=im_size, n_channels=n_channels)
    train_model.create_train_model()

    # init generate model
    generate_model = BEGAN(input_len=64, im_size=im_size, n_channels=n_channels)
    generate_model.create_generate_model()

    # create generator for sampling
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
                n_g_train=FLAGS.ng, n_d_train=FLAGS.nd, keep_prob=1.0,
                summary_writer=writer)
            generator.random_sampling(sess, plot_size=10, file_id=epoch_id)
            generator.viz_interpolate(sess, file_id=epoch_id)
            saver.save(sess,'{}gan-{}-epoch-{}'.format(save_path, FLAGS.gan_type, epoch_id))
        saver.save(sess, '{}gan-{}-epoch-{}'.format(save_path, FLAGS.gan_type, epoch_id))


if __name__ == "__main__":
    FLAGS = get_args()

    if FLAGS.train:
        train()
    if FLAGS.test:
        test()



#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: infogan.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from src.models.base import GANBaseModel
import src.models.layers as L
import src.models.modules as modules
import src.models.losses as losses


INIT_W = tf.random_normal_initializer(stddev=0.02)

class infoGAN(GANBaseModel):
    """ class for infoGAN """
    def __init__(self, input_len, im_size, n_channels,
                 n_continuous=0, n_discrete=0, mutual_info_weight=1.0):
        """
        Args:
            input_len (int): length of input random vector
            im_size (int or list with length 2): size of generate image 
            n_channels (int): number of image channels
            n_latent (int): number of structured latent variables
            mutual_info_weight (float): weight for mutual information regularization term
        """
        im_size = L.get_shape2D(im_size)
        self.in_len = input_len
        self.im_h, self.im_w = im_size
        self.n_channels = n_channels
        assert n_discrete >= 0 and n_continuous >= 0
        self.n_continuous = n_continuous
        self.n_discrete = n_discrete
        self.n_latent = n_continuous + n_discrete
        self._lambda = mutual_info_weight

        self.layers = {}

    def _create_train_input(self):
        """ input for training """
        self.random_vec = tf.placeholder(tf.float32, [None, self.in_len], 'input')
        self.real = tf.placeholder(
            tf.float32, 
            [None, self.im_h, self.im_w, self.n_channels],
            name='real')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def  _create_generate_input(self):
        """ input for sampling """
        self.random_vec = tf.placeholder(tf.float32, [None, self.in_len], 'input')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def create_train_model(self):
        """ create graph for training """
        self.set_is_training(True)
        self._create_train_input()
        fake = self.generator(self.random_vec)
        self.layers['generate'] = (fake + 1) / 2.
        self.layers['d_fake'] = self.discriminator(fake)
        self.layers['d_real'] = self.discriminator(self.real)

    def create_generate_model(self):
        """ create graph for sampling """
        self.set_is_training(False)
        self._create_generate_input()
        fake = self.generator(self.random_vec)
        self.layers['generate'] = (fake + 1) / 2.

    def _get_generator_loss(self):
        return losses.generator_cross_entropy_loss(self.layers['d_fake'])

    def _get_discriminator_loss(self):
        return losses.discriminator_cross_entropy_loss(
            self.layers['d_fake'], self.layers['d_real'])

    def _get_generator_optimizer(self):
        return tf.train.AdamOptimizer(self.lr, beta1=0.5)

    def _get_discriminator_optimizer(self):
        return tf.train.AdamOptimizer(self.lr, beta1=0.5)

    def generator(self, inputs):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            g_out = DCGAN_generator(
                inputs=inputs,
                init_w=INIT_W,
                is_training=self.is_training,
                layer_dict=self.layers,
                self.im_h,
                self.im_w,
                self.n_channels,
                final_dim=64,
                filter_size=5,
                wd=0,
                self.keep_prob=1.0)

            return g_out

    def discriminator(self, inputs):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            d_out = modules.DCGAN_discriminator(
                inputs=inputs,
                init_w=INIT_W,
                is_training=self.is_training,
                layer_dict=self.layers,
                start_depth=64,
                wd=0)

            D_conv = self.layers['DCGAN_discriminator_conv']

            c_mean, c_sigma = diagonal_Gaussian_layer(
                inputs=D_conv,
                layer_dict=self.layers,
                n_dim=self.n_latent,
                init_w=INIT_W,
                wd=0,
                is_training=self.is_training)

            return d_out

    def get_train_summary(self):
        with tf.name_scope('train'):
            tf.summary.image(
                'real_image',
                tf.cast((self.real + 1) / 2., tf.float32),
                collections=['train'])
            tf.summary.image(
                'generate_image',
                tf.cast(self.layers['generate'], tf.float32),
                collections=['train'])
        
        return tf.summary.merge_all(key='train')


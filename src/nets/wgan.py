#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: wgan.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from src.models.base import GANBaseModel
import src.models.layers as L
import src.models.modules as modules
import src.models.losses as losses


INIT_W = tf.random_normal_initializer(stddev=0.02)

class WGAN(GANBaseModel):
    """ class for WGAN """
    def __init__(self, input_len, im_size, n_channels):
        """
        Args:
            input_len (int): length of input random vector
            im_size (int or list with length 2): size of generate image 
            n_channels (int): number of image channels
        """
        im_size = L.get_shape2D(im_size)
        self.in_len = input_len
        self.im_h, self.im_w = im_size
        self.n_channels = n_channels

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

        bsize = tf.shape(fake)[0]
        e = tf.random_uniform(shape=[bsize, 1, 1, 1], minval=0, maxval=1., dtype=tf.float32, name='random_e')
        # e = tf.tile(e, [1, self.im_h, self.im_w, self.n_channels], name='tile_e')
        self.interp = e * self.real + (1 - e) * fake
        self.layers['d_interp'] = self.discriminator(self.interp)

        # self.train_d_op = self.get_discriminator_train_op(moniter=False)
        # self.train_g_op = self.get_generator_train_op(moniter=False)
        # self.d_loss_op = self.get_discriminator_loss()
        # self.g_loss_op = self.get_generator_loss()
        # self.train_summary_op = self.get_train_summary()

    def create_generate_model(self):
        """ create graph for sampling """
        self.set_is_training(False)
        self._create_generate_input()
        fake = self.generator(self.random_vec)
        self.layers['generate'] = (fake + 1) / 2.

    def _get_generator_loss(self):
        w_dist_loss = -tf.reduce_mean(self.layers['d_fake'])
        return w_dist_loss

    def _get_discriminator_loss(self):
        w_dist_loss = tf.reduce_mean(self.layers['d_fake']) - tf.reduce_mean(self.layers['d_real'])

        interp_grads = tf.gradients(self.layers['d_interp'], [self.interp])[0]
        grads_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(interp_grads), reduction_indices=[1]))
        grads_penalty = tf.reduce_mean((grads_l2_norm - 1.)**2)

        return w_dist_loss + 10 * grads_penalty

    def _get_generator_optimizer(self):
        return tf.train.AdamOptimizer(self.lr, beta1=0., beta2=0.9)

    def _get_discriminator_optimizer(self):
        return tf.train.AdamOptimizer(self.lr, beta1=0., beta2=0.9)

    def generator(self, inputs):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            self.layers['cur_input'] = inputs
            final_dim = 64
            filter_size = 3

            d_height_2, d_width_2 = L.deconv_size(self.im_h, self.im_w)
            d_height_4, d_width_4 = L.deconv_size(d_height_2, d_width_2)
            d_height_8, d_width_8 = L.deconv_size(d_height_4, d_width_4)
            d_height_16, d_width_16 = L.deconv_size(d_height_8, d_width_8)

            L.linear(out_dim=d_height_16 * d_width_16 * final_dim * 8,
                     layer_dict=self.layers,
                     init_w=INIT_W,
                     wd=0,
                     bn=True,
                     is_training=self.is_training,
                     name='Linear',
                     nl=tf.nn.relu)
            self.layers['cur_input'] = tf.reshape(
                self.layers['cur_input'],
                [-1, d_height_16, d_width_16, final_dim * 8])

            arg_scope = tf.contrib.framework.arg_scope
            with arg_scope([L.residual_block_iwgan], 
                            filter_size=filter_size, layer_dict=self.layers,
                            init_w=INIT_W, is_training=self.is_training, resample='up'):

                L.residual_block_iwgan(out_dim=final_dim * 8, name='res_block_1')
                L.residual_block_iwgan(out_dim=final_dim * 4, name='res_block_2')
                L.residual_block_iwgan(out_dim=final_dim * 2, name='res_block_3')
                L.residual_block_iwgan(out_dim=final_dim * 1, name='res_block_4')

            residual_out = self.layers['cur_input']
            residual_out = L.batch_norm(residual_out, train=self.is_training, name='residual_out_bn')
            residual_out = tf.nn.relu(residual_out, name='residual_out_relu')

            L.conv(
                filter_size=filter_size, out_dim=self.n_channels, layer_dict=self.layers,
                inputs=residual_out, bn=False, nl=tf.tanh,  init_w=INIT_W,
                is_training=self.is_training, name='out_conv')

            return self.layers['cur_input']

    def discriminator(self, inputs):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            self.layers['cur_input'] = inputs
            start_dim = 64
            filter_size = 3

            L.conv(
                filter_size=filter_size, out_dim=start_dim, layer_dict=self.layers,
                bn=False,  init_w=INIT_W,
                is_training=self.is_training, name='input_conv')

            arg_scope = tf.contrib.framework.arg_scope
            with arg_scope([L.residual_block_iwgan], 
                            filter_size=filter_size, layer_dict=self.layers,
                            init_w=INIT_W, is_training=self.is_training, resample='down'):

                L.residual_block_iwgan(out_dim=start_dim * 2, name='res_block_1')
                L.residual_block_iwgan(out_dim=start_dim * 4, name='res_block_2')
                L.residual_block_iwgan(out_dim=start_dim * 8, name='res_block_3')
                L.residual_block_iwgan(out_dim=start_dim * 8, name='res_block_4')

            bsize = tf.shape(inputs)[0]
            self.layers['cur_input'] = tf.reshape(
                self.layers['cur_input'], [bsize, 4, 4, start_dim * 8])
            L.linear(out_dim=1,
                     layer_dict=self.layers,
                     init_w=INIT_W,
                     wd=0,
                     bn=False,
                     is_training=self.is_training,
                     name='Linear')

            return self.layers['cur_input']

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



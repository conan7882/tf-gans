#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: lsgan.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from src.models.base import GANBaseModel
import src.models.layers as L
import src.models.modules as modules
import src.models.losses as losses


INIT_W = tf.random_normal_initializer(stddev=0.02)

class LSGAN(GANBaseModel):
    def __init__(self, input_len, im_size, n_channels):
        im_size = L.get_shape2D(im_size)
        self.in_len = input_len
        self.im_h, self.im_w = im_size
        self.n_channels = n_channels

        self.layers = {}

    def _create_train_input(self):
        self.random_vec = tf.placeholder(tf.float32, [None, self.in_len], 'input')
        self.real = tf.placeholder(
            tf.float32, 
            [None, self.im_h, self.im_w, self.n_channels],
            name='real')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def  _create_generate_input(self):
        self.random_vec = tf.placeholder(tf.float32, [None, self.in_len], 'input')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def create_train_model(self):
        self.set_is_training(True)
        self._create_train_input()
        fake = self.generator(self.random_vec)
        self.layers['generate'] = (fake + 1) / 2.
        self.layers['d_fake'] = self.discriminator(fake)
        self.layers['d_real'] = self.discriminator(self.real)

    def create_generate_model(self):
        self.set_is_training(False)
        self._create_generate_input()
        fake = self.generator(self.random_vec)
        self.layers['generate'] = (fake + 1) / 2.

    def _get_generator_loss(self):
        return losses.generator_least_square_loss(self.layers['d_fake'])

    def _get_discriminator_loss(self):
        return losses.discriminator_least_square_loss(
            self.layers['d_fake'], self.layers['d_real'])

    def _get_generator_optimizer(self):
        return tf.train.AdamOptimizer(self.lr, beta1=0.5)
        # return tf.train.RMSPropOptimizer(self.lr)

    def _get_discriminator_optimizer(self):
        return tf.train.AdamOptimizer(self.lr, beta1=0.5)
        # return tf.train.RMSPropOptimizer(self.lr)

    def generator(self, inputs):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            self.layers['cur_input'] = inputs
            # final_dim = 64
            # filter_size = 5
            b_size = tf.shape(inputs)[0]

            d_height_2, d_width_2 = L.deconv_size(self.im_h, self.im_w)
            d_height_4, d_width_4 = L.deconv_size(d_height_2, d_width_2)
            d_height_8, d_width_8 = L.deconv_size(d_height_4, d_width_4)
            d_height_16, d_width_16 = L.deconv_size(d_height_8, d_width_8)

            L.linear(out_dim=d_height_16 * d_width_16 * 256,
                     layer_dict=self.layers,
                     init_w=INIT_W,
                     wd=0,
                     bn=True,
                     is_training=self.is_training,
                     name='Linear',
                     nl=tf.nn.relu)
            self.layers['cur_input'] = tf.reshape(
                self.layers['cur_input'],
                [-1, d_height_16, d_width_16, 256])

            arg_scope = tf.contrib.framework.arg_scope
            with arg_scope([L.transpose_conv], 
                            filter_size=3, layer_dict=self.layers,
                            init_w=INIT_W, wd=0, is_training=self.is_training):
                output_shape = [b_size, d_height_8, d_width_8, 256]
                L.transpose_conv(out_shape=output_shape, stride=2,
                                 bn=True, nl=tf.nn.relu, name='dconv1')
                L.transpose_conv(out_shape=output_shape, stride=1,
                                 bn=True, nl=tf.nn.relu, name='dconv2')
                L.drop_out(self.layers, self.is_training, keep_prob=self.keep_prob)

                output_shape = [b_size, d_height_4, d_width_4, 256]
                L.transpose_conv(out_shape=output_shape, stride=2,
                                 bn=True, nl=tf.nn.relu, name='dconv3')
                L.transpose_conv(out_shape=output_shape, stride=1,
                                 bn=True, nl=tf.nn.relu, name='dconv4')
                L.drop_out(self.layers, self.is_training, keep_prob=self.keep_prob)

                output_shape = [b_size, d_height_2, d_width_2, 128]
                L.transpose_conv(out_shape=output_shape, stride=2,
                                 bn=True, nl=tf.nn.relu, name='dconv5')

                output_shape = [b_size, self.im_h, self.im_w, 64]
                L.transpose_conv(out_shape=output_shape, stride=2,
                                 bn=True, nl=tf.nn.relu, name='dconv6')
                L.drop_out(self.layers, self.is_training, keep_prob=self.keep_prob)

                output_shape = [b_size, self.im_h, self.im_w, self.n_channels]
                L.transpose_conv(out_shape=output_shape, stride=1,
                                 bn=False, nl=tf.tanh, name='dconv7')

                return self.layers['cur_input']

    def discriminator(self, inputs):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            d_out = modules.DCGAN_discriminator(
                inputs=inputs,
                init_w=INIT_W,
                is_training=self.is_training,
                layer_dict=self.layers,
                start_depth=64,
                wd=0)
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

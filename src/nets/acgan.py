#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: acgan.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from src.models.base import GANBaseModel
import src.models.layers as L
import src.models.modules as modules
import src.models.losses as losses
from src.helper.trainer_module import simple_train_epoch


INIT_W = tf.random_normal_initializer(stddev=0.02)

class ACGAN(GANBaseModel):
    """ class for ACGAN """
    def __init__(self, input_len, im_size, n_channels, n_class):
        """
        Args:
            input_len (int): length of input random vector
            im_size (int or list with length 2): size of generate image 
            n_channels (int): number of image channels
            n_class (int): number of image classes
        """
        im_size = L.get_shape2D(im_size)
        self.in_len = input_len
        self.im_h, self.im_w = im_size
        self.n_channels = n_channels
        self.n_class = n_class

        self.layers = {}

    def _create_train_input(self):
        """ input for training """
        self.random_vec = tf.placeholder(tf.float32, [None, self.in_len], 'input')
        self.fake_label = tf.placeholder(tf.int64, [None], name='fake_label')
        self.fake_label_onehot = tf.one_hot(
            self.fake_label, self.n_class, name='fake_label_onehot')
        self.z_in = tf.concate(
            [self.random_vec, self.fake_label_onehot], axis=-1, name='z_in')

        self.real = tf.placeholder(
            tf.float32, 
            [None, self.im_h, self.im_w, self.n_channels],
            name='real')
        self.real_label = tf.placeholder(tf.int64, [None], name='real_label')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # def  _create_generate_input(self):
    #     """ input for sampling """
    #     self.random_vec = tf.placeholder(tf.float32, [None, self.in_len], 'input')
    #     self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def create_train_model(self):
        """ create graph for training """
        self.set_is_training(True)
        self._create_train_input()
        self.epoch_id = 0
        self.global_step = 0

        fake = self.generator(self.z_in)
        self.layers['generate'] = (fake + 1) / 2.
        self.layers['d_fake'], self.layers['cls_logits_fake'] = self.discriminator(fake)
        self.layers['d_real'], self.layers['cls_logits_real'] = self.discriminator(self.real)

        self.train_d_op = self.get_discriminator_train_op(moniter=True)
        self.train_g_op = self.get_generator_train_op(moniter=True)
        self.d_loss_op = self.D_gan_loss
        self.g_loss_op = self.G_gan_loss
        self.fake_cls_loss_op = self.get_cls_loss(name='real')
        self.fake_cls_loss_op = self.get_cls_loss(name='fake')
        self.train_summary_op = self.get_train_summary()

    # def create_generate_model(self):
    #     """ create graph for sampling """
    #     self.set_is_training(False)
    #     self._create_generate_input()
    #     fake = self.generator(self.random_vec)
    #     self.layers['generate'] = (fake + 1) / 2.

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
            g_out = modules.ACGAN_generator(
                inputs=inputs,
                init_w=INIT_W,
                is_training=self.is_training,
                layer_dict=self.layers,
                im_h=self.im_h,
                im_w=self.im_w,
                n_channels=self.n_channels,
                final_dim=96,
                filter_size=5,
                wd=0,
                keep_prob=1.0)
            return g_out          

    def discriminator(self, inputs):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            d_out = modules.ACGAN_discriminator(
                inputs=inputs,
                init_w=INIT_W,
                is_training=self.is_training,
                layer_dict=self.layers,
                out_dim=self.n_class + 1,
                start_depth=16,
                keep_prob=self.keep_prob,
                wd=0)

            D_out = d_out[:, 0]
            cls_out = d_out[:, 1:]
            return D_out, cls_out

    def _get_cls_loss(self, name):
        with tf.name_scope('cls_loss'):
            if name == 'real':
                with tf.name_scope('real_cls_loss'):
                    real_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.real_label,
                        logits=self.layers['cls_logits_real'],
                        name='real_cross_entropy')
                return real_cross_entropy
            elif name == 'fake':
                with tf.name_scope('real_cls_loss'):
                    fake_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=self.fake_label,
                        logits=self.layers['cls_logits_fake'],
                        name='fake_cross_entropy')
                return fake_cross_entropy

    def get_cls_loss(self, name):
        try:
            return self._real_cls_loss if name == 'real' else self._fake_cls_loss
        except AttributeError:
            self._real_cls_loss = self._get_cls_loss(name='real')
            self._fake_cls_loss = self._get_cls_loss(name='fake')
        return self._real_cls_loss if name == 'real' else self._fake_cls_loss

    def _get_generator_loss(self):
        self.G_gan_loss = losses.generator_cross_entropy_loss(self.layers['d_fake'])
        return self.G_gan_loss + self.get_cls_loss(name='fake')

    def _get_discriminator_loss(self):
        self.D_gan_loss = losses.discriminator_cross_entropy_loss(
            self.layers['d_fake'], self.layers['d_real'])
        return self.D_gan_loss + self.get_cls_loss(name='fake') + self.get_cls_loss(name='real')

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

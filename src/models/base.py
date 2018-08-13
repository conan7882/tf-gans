#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
from abc import abstractmethod


class GANBaseModel(object):
    """ Base model of GAN """

    def set_is_training(self, is_training=True):
        self.is_training = is_training

    def get_generator_loss(self):
        try:
            return self._g_loss
        except AttributeError:
            self._g_loss = self._get_generator_loss()
        return self._g_loss

    def _get_generator_loss(self):
        raise NotImplementedError()

    def get_generator_optimizer(self):
        try:
            return self._g_opt
        except AttributeError:
            self._g_opt = self._get_generator_optimizer()
        return self._g_opt

    def _get_generator_optimizer(self):
        raise NotImplementedError()

    def get_generator_train_op(self, moniter=False):
        with tf.name_scope('generator_train'):
            opt = self.get_generator_optimizer()
            loss = self.get_generator_loss()
            var_list = tf.trainable_variables(scope='generator')
            grads = tf.gradients(loss, var_list)
            if moniter:
                [tf.summary.histogram('generator_gradient/' + var.name, grad, 
                    collections=['train']) for grad, var in zip(grads, var_list)]
            return opt.apply_gradients(zip(grads, var_list))

    def get_discriminator_loss(self):
        try:
            return self._d_loss
        except AttributeError:
            self._d_loss = self._get_discriminator_loss()
        return self._d_loss

    def _get_discriminator_loss(self):
        raise NotImplementedError()

    def get_discriminator_optimizer(self):
        try:
            return self._d_opt
        except AttributeError:
            self._d_opt = self._get_discriminator_optimizer()
        return self._d_opt

    def _get_discriminator_optimizer(self):
        raise NotImplementedError()

    def get_discriminator_train_op(self, moniter=False):
        with tf.name_scope('discriminator_train'):
            opt = self.get_discriminator_optimizer()
            loss = self.get_discriminator_loss()
            var_list = tf.trainable_variables(scope='discriminator')
            grads = tf.gradients(loss, var_list)
            if moniter:
                [tf.summary.histogram('discriminator_gradient/' + var.name, grad, 
                    collections=['train']) for grad, var in zip(grads, var_list)]
            return opt.apply_gradients(zip(grads, var_list))

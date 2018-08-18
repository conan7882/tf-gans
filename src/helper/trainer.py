#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: trainer.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np
import tensorflow as tf

# import src.utils.viz as viz
import src.models.distributions as distributions
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

def display(global_step,
            step,
            scaler_sum_list,
            name_list,
            collection,
            summary_val=None,
            summary_writer=None,
            ):
    print('[step: {}]'.format(global_step), end='')
    for val, name in zip(scaler_sum_list, name_list):
        print(' {}: {:.4f}'.format(name, val * 1. / step), end='')
    print('')
    if summary_writer is not None:
        s = tf.Summary()
        for val, name in zip(scaler_sum_list, name_list):
            s.value.add(tag='{}/{}'.format(collection, name),
                        simple_value=val * 1. / step)
        summary_writer.add_summary(s, global_step)
        if summary_val is not None:
            summary_writer.add_summary(summary_val, global_step)

class Trainer(object):
    def __init__(self, train_model, train_data,
                 moniter_gradient=False,
                 init_lr=1e-3, save_path=None):

        self._save_path = save_path
        self._t_model = train_model
        self._train_data = train_data
        self._lr = init_lr

        self._train_d_op = train_model.get_discriminator_train_op(moniter=moniter_gradient)
        self._train_g_op = train_model.get_generator_train_op(moniter=moniter_gradient)
        self._d_loss_op = train_model.get_discriminator_loss()
        self._g_loss_op = train_model.get_generator_loss()

        self._train_summary_op = train_model.get_train_summary()

        self.global_step = 0
        self.epoch_id = 0

    def train_epoch(self, sess, n_g_train=1, n_d_train=1, keep_prob=1.0, summary_writer=None):
        assert int(n_g_train) > 0 and int(n_d_train) > 0
        display_name_list = ['d_loss', 'g_loss']
        cur_summary = None

        if self.epoch_id == 100:
            self._lr = self._lr / 10
        if self.epoch_id == 300:
            self._lr = self._lr / 10

        cur_epoch = self._train_data.epochs_completed

        step = 0
        d_loss_sum = 0
        g_loss_sum = 0
        self.epoch_id += 1
        while cur_epoch == self._train_data.epochs_completed:
            self.global_step += 1
            step += 1

            batch_data = self._train_data.next_batch_dict()
            im = batch_data['im']
            # label = batch_data['label']
            
            # train discriminator
            for i in range(int(n_d_train)):
                random_vec = distributions.random_vector(
                    (len(im), self._t_model.in_len), dist_type='uniform')
                # random_vec = np.random.normal(
                #     size=(len(im), self._t_model.in_len))
                _, d_loss = sess.run(
                    [self._train_d_op, self._d_loss_op], 
                    feed_dict={self._t_model.real: im,
                               self._t_model.lr: self._lr,
                               self._t_model.keep_prob: keep_prob,
                               self._t_model.random_vec: random_vec})

            # train generator
            for i in range(int(n_g_train)):
                random_vec = distributions.random_vector(
                    (len(im), self._t_model.in_len), dist_type='uniform')
                # random_vec = np.random.normal(
                #     size=(len(im), self._t_model.in_len))
                _, g_loss = sess.run(
                    [self._train_g_op, self._g_loss_op], 
                    feed_dict={
                               self._t_model.lr: self._lr,
                               self._t_model.keep_prob: keep_prob,
                               self._t_model.random_vec: random_vec})

            d_loss_sum += d_loss
            g_loss_sum += g_loss

            if step % 100 == 0:
                cur_summary = sess.run(
                    self._train_summary_op, 
                    feed_dict={self._t_model.real: im,
                               self._t_model.keep_prob: keep_prob,
                               self._t_model.random_vec: random_vec})

                display(self.global_step,
                    step,
                    [d_loss_sum / n_d_train, g_loss_sum / n_g_train],
                    display_name_list,
                    'train',
                    summary_val=cur_summary,
                    summary_writer=summary_writer)

        print('==== epoch: {}, lr:{} ===='.format(cur_epoch, self._lr))
        cur_summary = sess.run(
                    self._train_summary_op, 
                    feed_dict={self._t_model.real: im,
                               self._t_model.keep_prob: keep_prob,
                               self._t_model.random_vec: random_vec})
        display(self.global_step,
                step,
                [d_loss_sum / n_d_train, g_loss_sum / n_g_train],
                display_name_list,
                'train',
                summary_val=cur_summary,
                summary_writer=summary_writer)

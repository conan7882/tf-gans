#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: trainer.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np
import tensorflow as tf

import src.models.distributions as distributions


def display(global_step, step, scaler_sum_list,
            name_list, collection,
            summary_val=None, summary_writer=None):
    """ Display averaged intermediate results for a period during training.

    The intermediate result will be displayed as:
    [step: global_step] name_list[0]: scaler_sum_list[0]/step ...
    Those result will be saved as summary as well.

    Args:
        global_step (int): index of current iteration
        step (int): number of steps for this period
        scaler_sum_list (float): list of summation of the intermediate
            results for this period
        name_list (str): list of display name for each intermediate result
        collection (str): list of graph collections keys for summary
        summary_val : additional summary to be saved
        summary_writer (tf.FileWriter): write for summary. No summary will be
            saved if None.
    """
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
    """ class for simple GAN training """
    def __init__(self, train_model, train_data,
                 moniter_gradient=False,
                 init_lr=1e-3, save_path=None):
        """ 
        Args:
            train_model (GANBaseModel): GAN model for traing
            train_data (DataFlow): dataflow for training set
            moniter_gradient (bool): save summary for gradients or not
            init_lr (float): initial learning rate
            save_path (str): Path to save summary. No summary will be
                saved if None.
        """

        self._save_path = save_path
        self._t_model = train_model
        self._train_data = train_data
        self._lr = init_lr

        self._train_d_op = train_model.get_discriminator_train_op(
            moniter=moniter_gradient)
        self._train_g_op = train_model.get_generator_train_op(
            moniter=moniter_gradient)
        self._d_loss_op = train_model.get_discriminator_loss()
        self._g_loss_op = train_model.get_generator_loss()

        try:
            self._update_op = train_model.update_k()
        except AttributeError:
            pass

        self._train_summary_op = train_model.get_train_summary()

        self.global_step = 0
        self.epoch_id = 0

    def train_epoch(self, sess, n_g_train=1, n_d_train=1, keep_prob=1.0,
                    summary_writer=None):
        """ Train for one epoch of training data

        Args:
            sess (tf.Session): tensorflow session
            n_g_train (int): number of times of generator training for each step
            n_d_train (int): number of times of discriminator training for each step
            keep_prob (float): keep probability for dropout
            summary_writer (tf.FileWriter): write for summary. No summary will be
            saved if None.
        """
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
            
            # train discriminator
            for i in range(int(n_d_train)):
                random_vec = distributions.random_vector(
                    (len(im), self._t_model.in_len), dist_type='uniform')
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

    # def train_epoch_2(self, sess, n_g_train=1, n_d_train=1, keep_prob=1.0, summary_writer=None):
    #     assert int(n_g_train) > 0 and int(n_d_train) > 0
    #     display_name_list = ['d_loss', 'g_loss']
    #     cur_summary = None

    #     if self.epoch_id == 100:
    #         self._lr = self._lr / 10
    #     if self.epoch_id == 300:
    #         self._lr = self._lr / 10

    #     cur_epoch = self._train_data.epochs_completed

    #     step = 0
    #     d_loss_sum = 0
    #     g_loss_sum = 0
    #     self.epoch_id += 1
    #     while cur_epoch == self._train_data.epochs_completed:
    #         self.global_step += 1
    #         step += 1

    #         batch_data = self._train_data.next_batch_dict()
    #         im = batch_data['im']
    #         # label = batch_data['label']


            
    #         # train discriminator
    #         for i in range(int(n_d_train)):
    #             random_vec = distributions.random_vector(
    #                 (len(im), self._t_model.in_len), dist_type='uniform')
    #             # random_vec = np.random.normal(
    #             #     size=(len(im), self._t_model.in_len))
    #             _, d_loss = sess.run(
    #                 [self._train_d_op, self._d_loss_op], 
    #                 feed_dict={self._t_model.real: im,
    #                            self._t_model.lr: self._lr,
    #                            self._t_model.keep_prob: keep_prob,
    #                            self._t_model.random_vec: random_vec})

    #         # train generator
    #         for i in range(int(n_g_train)):
    #             random_vec = distributions.random_vector(
    #                 (len(im), self._t_model.in_len), dist_type='uniform')
    #             # random_vec = np.random.normal(
    #             #     size=(len(im), self._t_model.in_len))
    #             _, g_loss = sess.run(
    #                 [self._train_g_op, self._g_loss_op], 
    #                 feed_dict={
    #                            self._t_model.lr: self._lr,
    #                            self._t_model.keep_prob: keep_prob,
    #                            self._t_model.random_vec: random_vec})

    #         random_vec = distributions.random_vector(
    #             (len(im), self._t_model.in_len), dist_type='uniform')
    #             # random_vec = np.random.normal(
    #             #     size=(len(im), self._t_model.in_len))
    #         sess.run(
    #             self._update_op, 
    #             feed_dict={self._t_model.real: im,
    #                        self._t_model.random_vec: random_vec})

            
            

    #         d_loss_sum += d_loss
    #         g_loss_sum += g_loss

    #         if step % 100 == 0:
    #             cur_summary = sess.run(
    #                 self._train_summary_op, 
    #                 feed_dict={self._t_model.real: im,
    #                            self._t_model.keep_prob: keep_prob,
    #                            self._t_model.random_vec: random_vec})

    #             display(self.global_step,
    #                 step,
    #                 [d_loss_sum / n_d_train, g_loss_sum / n_g_train],
    #                 display_name_list,
    #                 'train',
    #                 summary_val=cur_summary,
    #                 summary_writer=summary_writer)

    #     print('==== epoch: {}, lr:{} ===='.format(cur_epoch, self._lr))
    #     cur_summary = sess.run(
    #                 self._train_summary_op, 
    #                 feed_dict={self._t_model.real: im,
    #                            self._t_model.keep_prob: keep_prob,
    #                            self._t_model.random_vec: random_vec})
    #     display(self.global_step,
    #             step,
    #             [d_loss_sum / n_d_train, g_loss_sum / n_g_train],
    #             display_name_list,
    #             'train',
    #             summary_val=cur_summary,
    #             summary_writer=summary_writer)

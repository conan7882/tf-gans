#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: trainer_module.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

import src.utils.viz as viz
import src.models.distributions as distributions

def identity_feed(input_feed_dict):
    return input_feed_dict

def simple_train_epoch(train_model, sess, train_data, init_lr,
                       epoch_id, global_step, add_feed_fnc=identity_feed,
                       n_g_train=1, n_d_train=1, keep_prob=1.0,
                       summary_writer=None):
    """ Train for one epoch of training data

    Args:
        sess (tf.Session): tensorflow session
        train_model (GANBaseModel): GAN model for training
        train_data (DataFlow): DataFlow for training set
        init_lr (float): initial learning rate
        n_g_train (int): number of times of generator training for each step
        n_d_train (int): number of times of discriminator training for each step
        keep_prob (float): keep probability for dropout
        summary_writer (tf.FileWriter): write for summary. No summary will be
        saved if None.
    """

    assert int(n_g_train) > 0 and int(n_d_train) > 0
    display_name_list = ['d_loss', 'g_loss']
    cur_summary = None

    if epoch_id == 100:
        lr = init_lr / 10
    if epoch_id == 300:
        lr = init_lr / 10

    cur_epoch = train_data.epochs_completed

    step = 0
    d_loss_sum = 0
    g_loss_sum = 0
    self.epoch_id += 1
    while cur_epoch == train_data.epochs_completed:
        self.global_step += 1
        step += 1

        batch_data = train_data.next_batch_dict()
        im = batch_data['im']
        
        # train discriminator
        for i in range(int(n_d_train)):
            random_vec = distributions.random_vector(
                (len(im), train_model.in_len), dist_type='uniform')
            feed_dict={train_model.real: im,
                       train_model.lr: lr,
                       train_model.keep_prob: keep_prob,
                       train_model.random_vec: random_vec}
            feed_dict = add_feed_fnc(feed_dict)

            _, d_loss = sess.run(
                [train_model.train_d_op, train_model.d_loss_op],
                feed_dict=feed_dict,
                )
        # train generator
        for i in range(int(n_g_train)):
            random_vec = distributions.random_vector(
                (len(im), train_model.in_len), dist_type='uniform')
            feed_dict={train_model.lr: lr,
                       train_model.keep_prob: keep_prob,
                       train_model.random_vec: random_vec}
            feed_dict = add_feed_fnc(feed_dict)

            _, g_loss = sess.run(
                [train_model.train_g_op, train_model.g_loss_op], 
                feed_dict=feed_dict)

        d_loss_sum += d_loss
        g_loss_sum += g_loss

        if step % 100 == 0:
            feed_dict={train_model.real: im,
                       train_model.keep_prob: keep_prob,
                       train_model.random_vec: random_vec}
            feed_dict = add_feed_fnc(feed_dict)

            cur_summary = sess.run(
                self._train_summary_op, 
                feed_dict=feed_dict)

            viz.display(
                global_step,
                step,
                [d_loss_sum / n_d_train, g_loss_sum / n_g_train],
                display_name_list,
                'train',
                summary_val=cur_summary,
                summary_writer=summary_writer)

    print('==== epoch: {}, lr:{} ===='.format(cur_epoch, lr))
    feed_dict={train_model.real: im,
               train_model.keep_prob: keep_prob,
               train_model.random_vec: random_vec}
    feed_dict = add_feed_fnc(feed_dict)
    
    cur_summary = sess.run(
        train_model.train_summary_op, 
        feed_dict=feed_dict)
    viz.display(global_step,
        step,
        [d_loss_sum / n_d_train, g_loss_sum / n_g_train],
        display_name_list,
        'train',
        summary_val=cur_summary,
        summary_writer=summary_writer)

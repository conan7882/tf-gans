#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: began.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf

from src.models.base import GANBaseModel
import src.models.layers as L
import src.models.modules as modules
import src.models.losses as losses
import src.models.distributions as distributions
import src.utils.viz as viz


INIT_W = tf.random_normal_initializer(stddev=0.002)
BN = True

class BEGAN(GANBaseModel):
    def __init__(self, input_len, im_size, n_channels, gamma=0.6, lambda_k=1e-3):
        im_size = L.get_shape2D(im_size)
        # self.in_len = input_len
        self.im_h, self.im_w = im_size
        self.n_channels = n_channels
        self.n_code = input_len
        self.in_len = input_len
        self._lambda = lambda_k
        self._gamma = gamma

        self._decoder_start_size = 8

        self.layers = {}

    def _create_train_input(self):
        self.random_vec = tf.placeholder(tf.float32, [None, self.n_code], 'input')
        self.real = tf.placeholder(
            tf.float32, 
            [None, self.im_h, self.im_w, self.n_channels],
            name='real')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def  _create_generate_input(self):
        self.random_vec = tf.placeholder(tf.float32, [None, self.n_code], 'input')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def create_train_model(self):
        self.set_is_training(True)
        self._create_train_input()
        self.epoch_id = 0
        self.global_step = 0
        with tf.variable_scope('kt', reuse=tf.AUTO_REUSE):
            self.kt = tf.get_variable('kt',
                                      dtype=tf.float32,
                                      initializer=tf.constant(0.),
                                      trainable=True)
        self.fake = self.generator(self.random_vec)
        self.layers['generate'] = self.fake #(self.fake + 1) / 2.

        self.layers['decoder_fake'] = self.discriminator(self.fake)
        self.layers['decoder_real'] = self.discriminator(self.real)
        self.layers['convergence'] = self.get_convergence()

        self.train_d_op = self.get_discriminator_train_op(moniter=False)
        self.train_g_op = self.get_generator_train_op(moniter=False)
        self.d_loss_op = self.get_discriminator_loss()
        self.g_loss_op = self.get_generator_loss()
        self.update_op = self.update_k()
        self.train_summary_op = self.get_train_summary()

    def create_generate_model(self):
        self.set_is_training(False)
        self._create_generate_input()

        with tf.variable_scope('kt', reuse=tf.AUTO_REUSE):
            self.kt = tf.get_variable('kt',
                                      dtype=tf.float32,
                                      initializer=tf.constant(0.),
                                      trainable=True)

        self.fake = self.generator(self.random_vec)
        self.layers['generate'] = self.fake

    def update_k(self):
        with tf.name_scope('update_k'):
            real_x = self.real
            real_y = self.layers['decoder_real']
            fake_x = self.fake
            fake_y = self.layers['decoder_fake']
            L_real = losses.l2_loss(real_x, real_y)
            L_fake = losses.l2_loss(fake_x, fake_y)
            new_k = self.kt + self._lambda * (self._gamma * L_real - L_fake)
            new_k = tf.clip_by_value(new_k, 0., 1., name='clip_k')
            update = self.kt.assign(new_k)
            return update

    def get_convergence(self):
        with tf.name_scope('convergence'):
            real_x = self.real
            real_y = self.layers['decoder_real']
            fake_x = self.fake
            fake_y = self.layers['decoder_fake']
            L_real = losses.l2_loss(real_x, real_y)
            L_fake = losses.l2_loss(fake_x, fake_y)
            return L_real + tf.abs(self._gamma * L_real - L_fake)

    def _get_generator_loss(self):
        with tf.name_scope('generator_loss'):
            x = self.fake
            y = self.layers['decoder_fake']
            return losses.l2_loss(x, y)

    def _get_discriminator_loss(self):
        with tf.name_scope('discriminator_loss'):
            real_x = self.real
            real_y = self.layers['decoder_real']
            fake_x = self.fake
            fake_y = self.layers['decoder_fake']

            return losses.l2_loss(real_x, real_y)\
                - self.kt * losses.l2_loss(fake_x, fake_y)

    def _get_generator_optimizer(self):
        return tf.train.AdamOptimizer(self.lr, beta1=0.5)

    def _get_discriminator_optimizer(self):
        return tf.train.AdamOptimizer(self.lr, beta1=0.5)

    def generator(self, inputs):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):

            # modules.BEGAN_decoder(
            #     inputs=inputs,
            #     layer_dict=self.layers,
            #     start_size=self._decoder_start_size,
            #     n_feature=64,
            #     n_channle=self.n_channels,
            #     init_w=INIT_W,
            #     is_training=self.is_training,
            #     bn=BN,
            #     wd=0)

            decoder_out = modules.LSGAN_generator(
                inputs=inputs,
                layer_dict=self.layers,
                im_size=[self.im_h, self.im_w],
                n_channle=self.n_channels,
                init_w=INIT_W,
                keep_prob=self.keep_prob,
                wd=0,
                is_training=self.is_training,
                bn=BN)
            
            output = self.layers['cur_input']
            output = tf.reshape(
                output, [-1, self.im_h, self.im_w, self.n_channels])

            return output

    def discriminator(self, inputs):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            encoder_out = modules.BEGAN_encoder(
                inputs=inputs,
                layer_dict=self.layers,
                n_code=self.n_code,
                start_depth=64,
                init_w=INIT_W,
                is_training=self.is_training,
                bn=BN,
                wd=0)

            decoder_out = modules.LSGAN_generator(
                inputs=encoder_out,
                layer_dict=self.layers,
                im_size=[self.im_h, self.im_w],
                n_channle=self.n_channels,
                init_w=INIT_W,
                keep_prob=self.keep_prob,
                wd=0,
                is_training=self.is_training,
                bn=BN)

            # decoder_out = modules.BEGAN_decoder(
            #     inputs=encoder_out,
            #     layer_dict=self.layers,
            #     start_size=self._decoder_start_size,
            #     n_feature=64,
            #     n_channle=self.n_channels,
            #     init_w=INIT_W,
            #     is_training=self.is_training,
            #     bn=BN,
            #     wd=0)

            return decoder_out

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
            tf.summary.image(
                'decoder_real_image',
                tf.cast(self.layers['decoder_real'], tf.float32),
                collections=['train'])
            tf.summary.scalar('kt', self.kt, collections=['train'])
            tf.summary.scalar('convergence', self.layers['convergence'],
                              collections=['train'])
        return tf.summary.merge_all(key='train')

    def train_epoch(self, sess, train_data, init_lr,
                    n_g_train=1, n_d_train=1, keep_prob=1.0,
                    summary_writer=None):

        assert int(n_g_train) > 0 and int(n_d_train) > 0
        display_name_list = ['d_loss', 'g_loss']
        cur_summary = None

        lr = init_lr
        # if self.epoch_id == 100:
        #     lr = init_lr / 10
        # if self.epoch_id == 300:
        #     lr = init_lr / 10

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
                    (len(im), self.n_code), dist_type='uniform')
                # random_vec = np.random.normal(
                #     size=(len(im), self._t_model.in_len))
                _, d_loss = sess.run(
                    [self.train_d_op, self.d_loss_op], 
                    feed_dict={self.real: im,
                               self.lr: lr,
                               self.keep_prob: keep_prob,
                               self.random_vec: random_vec})

            # train generator
            for i in range(int(n_g_train)):
                random_vec = distributions.random_vector(
                    (len(im), self.n_code), dist_type='uniform')
                _, g_loss = sess.run(
                    [self.train_g_op, self.g_loss_op], 
                    feed_dict={
                               self.lr: lr,
                               self.keep_prob: keep_prob,
                               self.random_vec: random_vec})

            random_vec = distributions.random_vector(
                    (len(im), self.n_code), dist_type='uniform')
            sess.run(
                self.update_op, 
                feed_dict={self.real: im,
                           self.random_vec: random_vec})

            d_loss_sum += d_loss
            g_loss_sum += g_loss

            if step % 100 == 0:
                cur_summary = sess.run(
                    self.train_summary_op, 
                    feed_dict={self.real: im,
                               self.keep_prob: keep_prob,
                               self.random_vec: random_vec})

                viz.display(
                    self.global_step,
                    step,
                    [d_loss_sum / n_d_train, g_loss_sum / n_g_train],
                    display_name_list,
                    'train',
                    summary_val=cur_summary,
                    summary_writer=summary_writer)

        print('==== epoch: {}, lr:{} ===='.format(cur_epoch, lr))
        cur_summary = sess.run(
                    self.train_summary_op, 
                    feed_dict={self.real: im,
                               self.keep_prob: keep_prob,
                               self.random_vec: random_vec})
        viz.display(
            self.global_step,
            step,
            [d_loss_sum / n_d_train, g_loss_sum / n_g_train],
            display_name_list,
            'train',
            summary_val=cur_summary,
            summary_writer=summary_writer)


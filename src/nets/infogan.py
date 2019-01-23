#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: infogan.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import math
import tensorflow as tf
import numpy as np

from src.models.base import GANBaseModel
import src.models.layers as L
import src.models.modules as modules
import src.models.entropies as entropies
import src.models.losses as losses
import src.utils.utils as utils
import src.models.distributions as distributions
import src.utils.viz as viz
import src.utils.dataflow as dfutils


INIT_W = tf.random_normal_initializer(stddev=0.02)

class infoGAN(GANBaseModel):
    """ class for infoGAN """
    def __init__(self, input_len, im_size, n_channels,
                 cat_n_class_list, n_continuous=0, n_discrete=0,
                 mutual_info_weight=1.0, max_grad_norm=10.):
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
        self.n_code = input_len
        assert n_discrete >= 0 and n_continuous >= 0
        self.n_continuous = n_continuous
        self.n_discrete = n_discrete
        self.cat_n_class_list = utils.make_list(cat_n_class_list)
        assert len(self.cat_n_class_list) >= self.n_discrete
        self.n_latent = n_continuous + n_discrete
        self._lambda = mutual_info_weight

        self._max_grad_norm = max_grad_norm
        self.layers = {}

    def _create_train_input(self):
        """ input for training """
        self.random_vec = tf.placeholder(tf.float32, [None, self.in_len], 'input')
        len_discrete_code = np.sum(self.cat_n_class_list[:self.n_discrete])

        self.code_discrete = tf.placeholder(
            tf.int64, [None, len_discrete_code], name='code_discrete')
        self.discrete_label = tf.placeholder(
            tf.int64, [None, self.n_discrete], name='discrete_label')

        self.code_continuous = tf.placeholder(
            tf.float32, [None, self.n_continuous], name='code_continuous')

        input_discrete = tf.cast(self.code_discrete, tf.float32)
        self.input_vec = tf.concat(
            (self.random_vec, input_discrete, self.code_continuous),
            axis=-1)

        self.real = tf.placeholder(
            tf.float32, 
            [None, self.im_h, self.im_w, self.n_channels],
            name='real')
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def  _create_generate_input(self):
        """ input for sampling """
        self.random_vec = tf.placeholder(tf.float32, [None, self.in_len], 'input')
        len_discrete_code = np.sum(self.cat_n_class_list[:self.n_discrete])
        self.code_discrete = tf.placeholder(
            tf.int64, [None, len_discrete_code], name='code_discrete')

        self.code_continuous = tf.placeholder(
            tf.float32, [None, self.n_continuous], name='code_continuous')

        input_discrete = tf.cast(self.code_discrete, tf.float32)
        self.input_vec = tf.concat(
            (self.random_vec, input_discrete, self.code_continuous),
            axis=-1)

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def create_train_model(self):
        """ create graph for training """
        self.set_is_training(True)
        self._create_train_input()
        self.epoch_id = 0
        self.global_step = 0
        fake = self.generator(self.input_vec)
        self.layers['generate'] = (fake + 1) / 2.
        self.layers['d_fake'], self.layers['Q_discrete_logits_fake'], self.layers['Q_cont_log_prob_list'] =\
            self.discriminator(fake)
        self.layers['d_real'], self.layers['Q_discrete_logits_real'], _ =\
            self.discriminator(self.real)

        self.train_d_op = self.get_discriminator_train_op(moniter=False)
        self.train_g_op = self.get_generator_train_op(moniter=False)
        self.d_loss_op = self.D_gan_loss
        self.g_loss_op = self.G_gan_loss
        self.train_summary_op = self.get_train_summary()

    def create_generate_model(self):
        """ create graph for sampling """
        self.set_is_training(False)
        self._create_generate_input()
        fake = self.generator(self.input_vec)
        self.layers['generate'] = (fake + 1) / 2.

        self.generate_op = self.layers['generate']

    def _get_total_entropy(self):
        total_entropy = 0
        for i in range(self.n_discrete):
            n_class = self.cat_n_class_list[i]
            total_entropy += entropies.entropy_uniform(n_class)

        total_entropy += entropies.entropy_isotropic_Gaussian(
            self.n_continuous, sigma=1.0)
        return total_entropy

    def get_total_entropy(self):
        try:
            return self.total_entropy
        except AttributeError:
            self.total_entropy = self._get_total_entropy()
            return self.total_entropy

    def _get_Q_discrete_loss(self):
        if self.n_discrete <= 0:
            return tf.constant(0.)
        code_discrete_list = tf.unstack(self.discrete_label, axis=-1)
        loss = 0
        for idx, (label, logits) in enumerate(zip(code_discrete_list, self.layers['Q_discrete_logits_fake'])):
            
            cur_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=label, logits=logits,
                name='discrete_loss_{}'.format(idx))
            loss += tf.reduce_mean(cur_loss)
        return loss

    def _get_Q_cont_loss(self):
        if self.n_continuous > 0:
            return -tf.reduce_mean(self.layers['Q_cont_log_prob_list'])
        else:
            return tf.constant(0.)

    def get_Q_loss(self):
        try:
            return self.Q_loss
        except AttributeError:
            self.Q_loss = self._get_Q_discrete_loss() + 0.1 * self._get_Q_cont_loss()
            return self.Q_loss

    def _get_generator_loss(self):
        self.G_gan_loss = losses.generator_cross_entropy_loss(self.layers['d_fake'])
        total_entropy = self.get_total_entropy()
        Q_fake_loss = self.get_Q_loss()
        info_loss = Q_fake_loss - total_entropy
        self.LI_G = -info_loss

        return self.G_gan_loss + self._lambda * info_loss

    def _get_discriminator_loss(self):
        self.D_gan_loss = losses.discriminator_cross_entropy_loss(
            self.layers['d_fake'], self.layers['d_real'])
        total_entropy = self.get_total_entropy()
        Q_fake_loss = self.get_Q_loss()
        info_loss = Q_fake_loss - total_entropy
        self.LI_D = -info_loss
        return self.D_gan_loss + self._lambda * info_loss

    def _get_generator_optimizer(self):
        return tf.train.AdamOptimizer(self.lr, beta1=0.5)

    def _get_discriminator_optimizer(self):
        return tf.train.AdamOptimizer(self.lr, beta1=0.5)

    def generator(self, inputs):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            g_out = modules.InfoGAN_MNIST_generator(
                inputs=inputs,
                init_w=INIT_W,
                is_training=self.is_training,
                layer_dict=self.layers,
                im_h=self.im_h,
                im_w=self.im_w,
                n_channels=self.n_channels,
                final_dim=64,
                filter_size=5,
                wd=0,
                keep_prob=self.keep_prob)

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

        with tf.variable_scope('sample_Q', reuse=tf.AUTO_REUSE):
            Q_discrete_logits_list = []
            for i in range(self.n_discrete):
                n_class = self.cat_n_class_list[i]
                Q_discrete_logits = modules.categorical_distribution_layer(
                        inputs=D_conv,
                        layer_dict=self.layers,
                        n_class=n_class,
                        init_w=INIT_W, wd=0, is_training=self.is_training,
                        name='Cat_{}'.format(i))
                Q_discrete_logits_list.append(Q_discrete_logits)

            Q_cont_log_prob_list = []
            if self.n_continuous > 0:
                c_mean, c_sigma = modules.diagonal_Gaussian_layer(
                    inputs=D_conv,
                    layer_dict=self.layers,
                    n_dim=self.n_continuous,
                    init_w=INIT_W,
                    wd=0,
                    is_training=self.is_training)

                Q_cont_log_prob_list = modules.evaluate_log_diagonal_Gaussian_pdf(
                    c_mean, c_sigma, self.code_continuous)

        return d_out, Q_discrete_logits_list, Q_cont_log_prob_list

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

    def get_discriminator_train_op(self, moniter=False):
        with tf.name_scope('discriminator_train'):
            opt = self.get_discriminator_optimizer()
            loss = self.get_discriminator_loss()
            var_list = tf.trainable_variables(scope='discriminator')\
                + tf.trainable_variables(scope='sample_Q')
            try:
                if self._max_grad_norm > 0:
                    grads, _ = tf.clip_by_global_norm(
                        tf.gradients(loss, var_list),
                        self._max_grad_norm)
                else:
                    grads = tf.gradients(loss, var_list)
            except AttributeError:
                grads = tf.gradients(loss, var_list)

            if moniter:
                [tf.summary.histogram('discriminator_gradient/' + var.name, grad, 
                    collections=['train']) for grad, var in zip(grads, var_list)]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = opt.apply_gradients(zip(grads, var_list))
            return train_op

    def train_epoch(self, sess, train_data, init_lr,
                    n_g_train=1, n_d_train=1, keep_prob=1.0,
                    summary_writer=None):
        """ Train for one epoch of training data

        Args:
            sess (tf.Session): tensorflow session
            train_data (DataFlow): DataFlow for training set
            init_lr (float): initial learning rate
            n_g_train (int): number of times of generator training for each step
            n_d_train (int): number of times of discriminator training for each step
            keep_prob (float): keep probability for dropout
            summary_writer (tf.FileWriter): write for summary. No summary will be
            saved if None.
        """

        assert int(n_g_train) > 0 and int(n_d_train) > 0
        display_name_list = ['d_loss', 'g_loss', 'LI_G', 'LI_D']
        cur_summary = None

        lr = init_lr
        lr_D = 2e-4
        lr_G = 1e-3
        # lr_G = lr_D * 10

        cur_epoch = train_data.epochs_completed
        step = 0
        d_loss_sum = 0
        g_loss_sum = 0
        LI_G_sum = 0
        LI_D_sum = 0
        self.epoch_id += 1
        while cur_epoch == train_data.epochs_completed:
            self.global_step += 1
            step += 1

            batch_data = train_data.next_batch_dict()
            im = batch_data['im']

            random_vec = distributions.random_vector(
                (len(im), self.n_code), dist_type='uniform')

            # code_discrete = []
            # discrete_label = []
            if self.n_discrete <= 0:
                code_discrete = np.zeros((len(im), 0))
                discrete_label = np.zeros((len(im), self.n_discrete))
            else:
                code_discrete = []
                discrete_label = []
                for i in range(self.n_discrete):
                    n_class = self.cat_n_class_list[i]
                    cur_code = np.random.choice(n_class, (len(im)))
                    cur_onehot_code = dfutils.vec2onehot(cur_code, n_class)
                    try:
                        code_discrete = np.concatenate((code_discrete, cur_onehot_code), axis=-1)
                        discrete_label = np.concatenate(
                            (discrete_label, np.expand_dims(cur_code, axis=-1)),
                            axis=-1)
                    except ValueError:
                        code_discrete = cur_onehot_code
                        discrete_label = np.expand_dims(cur_code, axis=-1)

            code_cont = distributions.random_vector(
                (len(im), self.n_continuous), dist_type='uniform')
            
            # train discriminator
            for i in range(int(n_d_train)):
                
                _, d_loss, LI_D = sess.run(
                    [self.train_d_op, self.d_loss_op, self.LI_D], 
                    feed_dict={self.real: im,
                               self.lr: lr_D,
                               self.keep_prob: keep_prob,
                               self.random_vec: random_vec,
                               self.code_discrete: code_discrete,
                               self.discrete_label: discrete_label,
                               self.code_continuous: code_cont})
            # train generator
            for i in range(int(n_g_train)):
                _, g_loss, LI_G = sess.run(
                    [self.train_g_op, self.g_loss_op, self.LI_G], 
                    feed_dict={
                               self.lr: lr_G,
                               self.keep_prob: keep_prob,
                               self.random_vec: random_vec,
                               self.code_discrete: code_discrete,
                               self.discrete_label: discrete_label,
                               self.code_continuous: code_cont})

            d_loss_sum += d_loss
            g_loss_sum += g_loss
            LI_G_sum += LI_G
            LI_D_sum += LI_D

            if step % 100 == 0:
                cur_summary = sess.run(
                    self.train_summary_op, 
                    feed_dict={self.real: im,
                               self.keep_prob: keep_prob,
                               self.random_vec: random_vec,
                               self.code_discrete: code_discrete,
                               self.discrete_label: discrete_label,
                               self.code_continuous: code_cont})

                viz.display(
                    self.global_step,
                    step,
                    [d_loss_sum / n_d_train, g_loss_sum / n_g_train,
                     LI_G_sum, LI_D_sum],
                    display_name_list,
                    'train',
                    summary_val=cur_summary,
                    summary_writer=summary_writer)

        print('==== epoch: {}, lr:{} ===='.format(cur_epoch, lr))
        cur_summary = sess.run(
            self.train_summary_op, 
            feed_dict={self.real: im,
                       self.keep_prob: keep_prob,
                       self.random_vec: random_vec,
                       self.code_discrete: code_discrete,
                       self.discrete_label: discrete_label,
                       self.code_continuous: code_cont})
        viz.display(
            self.global_step,
            step,
            [d_loss_sum / n_d_train, g_loss_sum / n_g_train,
             LI_G_sum, LI_D_sum],
            display_name_list,
            'train',
            summary_val=cur_summary,
            summary_writer=summary_writer)

    def generate_samples(self, sess, keep_prob=1.0, file_id=None, save_path=None):
        self.random_sampling(
            sess, keep_prob=keep_prob, plot_size=10,
            file_id=file_id, save_path=save_path)

        for vary_discrete_id in range(self.n_discrete):
            self.vary_discrete_sampling(
                sess, vary_discrete_id,
                keep_prob=keep_prob, sample_per_class=10,
                file_id=file_id, save_path=save_path)
        for cont_code_id in range(self.n_continuous):
            self.interp_cont_sampling(
                sess, n_interpolation=10, cont_code_id=cont_code_id,
                vary_discrete_id=0, n_col_samples=5,
                keep_prob=keep_prob, save_path=save_path, file_id=file_id)

    def random_sampling(self, sess,
                        keep_prob=1.0, plot_size=10,
                        file_id=None, save_path=None):
        n_samples = plot_size * plot_size
        random_vec = distributions.random_vector(
            (n_samples, self.n_code), dist_type='uniform')
        code_cont = distributions.random_vector(
            (n_samples, self.n_continuous), dist_type='uniform')

        if self.n_discrete <= 0:
            code_discrete = np.zeros((n_samples, 0))
        else:
            code_discrete = []
            for i in range(self.n_discrete):
                n_class = self.cat_n_class_list[i]
                cur_code = [np.random.choice(n_class) for i in range(n_samples)]
                cur_code = dfutils.vec2onehot(cur_code, n_class)
                try:
                    code_discrete = np.concatenate((code_discrete, cur_code), axis=-1)
                except ValueError:
                    code_discrete = cur_code

        self._viz_samples(
            sess, random_vec, code_discrete, code_cont, keep_prob,
            plot_size=[plot_size, plot_size], save_path=save_path,
            file_name='random_sampling', file_id=file_id)

    def vary_discrete_sampling(self, sess, vary_discrete_id,
                               keep_prob=1.0, sample_per_class=10,
                               file_id=None, save_path=None):
        """ Sampling by varying a discrete code.

        Args:
            sess (tf.Session): tensorflow session
            vary_discrete_id (int): index of discrete code for varying
            keep_prob (float): keep probability for dropout
            sample_per_class (int): number of samples for each class
            file_id (int): index for saving image 
            save_path (str): directory for saving image
        """
        if vary_discrete_id >= self.n_discrete:
            return
        n_vary_class = self.cat_n_class_list[vary_discrete_id]
        n_samples = n_vary_class * sample_per_class
        
        # sample_per_class = int(math.floor(n_samples / n_vary_class))
        n_remain_sample = n_samples - n_vary_class * sample_per_class

        random_vec = distributions.random_vector(
            (n_samples, self.n_code), dist_type='uniform')

        if self.n_discrete <= 0:
            code_discrete = np.zeros((n_samples, 0))
        else:
            code_discrete = []
            for i in range(self.n_discrete):
                n_class = self.cat_n_class_list[i]
                if i == vary_discrete_id:
                    cur_code = [i for i in range(n_class) for j in range(sample_per_class)]
                else:
                    cur_code = [np.random.choice(n_class)
                                for j in range(sample_per_class)]
                    cur_code = np.tile(cur_code, (n_vary_class))
                cur_code = dfutils.vec2onehot(cur_code, n_class)
                try:
                    code_discrete = np.concatenate((code_discrete, cur_code), axis=-1)
                except ValueError:
                    code_discrete = cur_code

        code_cont = distributions.random_vector(
            (n_samples, self.n_continuous), dist_type='uniform')

        self._viz_samples(
            sess, random_vec, code_discrete, code_cont, keep_prob,
            plot_size=[n_vary_class, sample_per_class], save_path=save_path,
            file_name='vary_discrete_{}'.format(vary_discrete_id), file_id=file_id)

    def interp_cont_sampling(self, sess, n_interpolation,
                             cont_code_id, vary_discrete_id=None, n_col_samples=None,
                             keep_prob=1., save_path=None, file_id=None):
        """ Sample interpolation of one of continuous codes.

        Args:
            sess (tf.Session): tensorflow session
            n_interpolation (int): number of interpolation samples
            cont_code_id (int): index of continuous code for interpolation
            vary_discrete_id (int): Index of discrete code for varying
                while sample interpolation. All the discrete code will be fixed
                if it is None.
            keep_prob (float): keep probability for dropout
            save_path (str): directory for saving image
            file_id (int): index for saving image 
        """
        if cont_code_id >= self.n_continuous:
            return
        if vary_discrete_id is not None and vary_discrete_id < self.n_discrete:
            n_vary_class = self.cat_n_class_list[vary_discrete_id]
            n_samples = n_interpolation * n_vary_class
        elif n_col_samples is not None:
            n_vary_class = n_col_samples
            n_samples = n_interpolation * n_vary_class
        else:
            n_vary_class = 1
            n_samples = n_interpolation


        random_vec = distributions.random_vector(
            (1, self.n_code), dist_type='uniform')
        random_vec = np.tile(random_vec, (n_samples, 1))

        if self.n_discrete <= 0:
            code_discrete = np.zeros((n_samples, 0))
        else:
            code_discrete = []
            for i in range(self.n_discrete):
                n_class = self.cat_n_class_list[i]
                if i == vary_discrete_id:
                    cur_code = [i for i in range(n_class) for j in range(n_interpolation)]
                else:
                    cur_code = np.random.choice(n_class, 1) * np.ones((n_samples))
                cur_onehot_code = dfutils.vec2onehot(cur_code, n_class)
                try:
                    code_discrete = np.concatenate(
                        (code_discrete, cur_onehot_code), axis=-1)
                except ValueError:
                    code_discrete = cur_onehot_code

        if vary_discrete_id is not None and vary_discrete_id < self.n_discrete:
            code_cont = distributions.random_vector(
                (1, self.n_continuous), dist_type='uniform')
            code_cont = np.tile(code_cont, (n_samples, 1))
            cont_interp = np.linspace(-1., 1., n_interpolation)
            cont_interp = np.tile(cont_interp, (n_vary_class))
            code_cont[:, cont_code_id] = cont_interp
        else:
            code_cont = distributions.random_vector(
                (n_col_samples, self.n_continuous), dist_type='uniform')
            code_cont = np.repeat(code_cont, n_interpolation, axis=0)
            # code_cont = np.tile(code_cont.transpose(), (1, n_interpolation)).transpose()
            cont_interp = np.linspace(-1., 1., n_interpolation)
            cont_interp = np.tile(cont_interp, (n_vary_class))
            code_cont[:, cont_code_id] = cont_interp

        self._viz_samples(
            sess, random_vec, code_discrete, code_cont, keep_prob,
            plot_size=[n_vary_class, n_interpolation], save_path=save_path,
            file_name='interp_cont_{}'.format(cont_code_id), file_id=file_id)

    def _viz_samples(self, sess, random_vec, code_discrete, code_cont, keep_prob,
                     plot_size=10, save_path=None, file_name='generate_im', file_id=None):
        """ Sample and save images from model as one single image.

        Args:
            sess (tf.Session): tensorflow session
            random_vec (float): list of input random vectors
            code_discrete (list): list of discrete code (one hot vectors)
            code_cont (list): list of continuous code
            keep_prob (float): keep probability for dropout
            plot_size (int): side size (number of samples) of saving image
            save_path (str): directory for saving image
            file_name (str): name for saving image
            file_id (int): index for saving image 
        """
        if save_path:
            plot_size = utils.get_shape2D(plot_size)
            gen_im = sess.run(self.generate_op,
                feed_dict={self.random_vec: random_vec,
                           self.keep_prob: keep_prob,
                           self.code_discrete: code_discrete,
                           self.code_continuous: code_cont})

            if file_id is not None:
                im_save_path = os.path.join(
                    save_path, '{}_{}.png'.format(file_name, file_id))
            else:
                im_save_path = os.path.join(
                    save_path, '{}.png'.format(file_name))

            viz.viz_batch_im(batch_im=gen_im, grid_size=plot_size,
                             save_path=im_save_path, gap=0, gap_color=0,
                             shuffle=False)







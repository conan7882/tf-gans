#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: generator.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import math
import numpy as np
import tensorflow as tf

import src.utils.viz as viz
import src.utils.utils as utils
import src.models.distributions as distributions


class Generator(object):
    def __init__(self, generate_model, save_path=None):

        self._save_path = save_path
        self._g_model = generate_model
        self._generate_op = generate_model.layers['generate']

    def random_sampling(self, sess, plot_size=10, file_id=None):
        n_samples = plot_size * plot_size
        random_vec = distributions.random_vector(
            (n_samples, self._g_model.in_len), dist_type='uniform')
        # random_vec = np.random.normal(
        #     size=(n_samples, self._g_model.in_len))
        if self._save_path:
            self._viz_samples(sess, random_vec, plot_size,
                              file_id=file_id)

    def viz_interpolate(self, sess, n_sample=5, n_interp=15, file_id=None):
        random_vec = distributions.linear_interpolate(
            z1=None, z2=None,
            z_shape=(n_sample, self._g_model.in_len),
            n_samples=n_interp)
        random_vec = np.transpose(random_vec, (1, 0, 2))
        random_vec = np.reshape(random_vec, (-1, self._g_model.in_len))
        if self._save_path:
            self._viz_samples(sess, random_vec, plot_size=[n_sample, n_interp],
                              file_name='interpolate', file_id=file_id)

    def viz_2D_manifold(self, sess, plot_size, file_id=None):
        # if z is None:
        #     gen_im = sess.run(self._generate_op)
        # else:
        n_samples = plot_size * plot_size

        random_vec = distributions.linear_2D_interpolate(
            side_size=20,interpolate_range=[-1, 1, -1, 1])
        if self._save_path:
            self._viz_samples(sess, random_vec, plot_size,
                              file_name='manifoid', file_id=file_id)

    def _viz_samples(self, sess, random_vec,
                     plot_size=10, file_name='generate_im', file_id=None):
        plot_size = utils.get_shape2D(plot_size)
        gen_im = sess.run(self._generate_op,
                          feed_dict={self._g_model.random_vec: random_vec})
        if self._save_path:
            if file_id is not None:
                im_save_path = os.path.join(
                    self._save_path, '{}_{}.png'.format(file_name, file_id))
            else:
                im_save_path = os.path.join(
                    self._save_path, '{}.png'.format(file_name))

            # n_sample = len(gen_im)
            # plot_size = int(min(plot_size, math.sqrt(n_sample)))
            viz.viz_batch_im(batch_im=gen_im, grid_size=plot_size,
                             save_path=im_save_path, gap=0, gap_color=0,
                             shuffle=False)



            


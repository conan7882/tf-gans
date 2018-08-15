#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: modules.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import src.models.layers as L

def DCGAN_discriminator(inputs, init_w, is_training, layer_dict, start_depth=64, wd=0):
    layer_dict['cur_input'] = inputs
    # start_depth = 64
    filter_size = 5
    b_size = tf.shape(inputs)[0]

    arg_scope = tf.contrib.framework.arg_scope
    with arg_scope([L.conv], 
                    filter_size=filter_size, layer_dict=layer_dict,
                    stride=2, nl=L.leaky_relu, add_summary=False,
                    init_w=init_w, wd=0, is_training=is_training):

        L.conv(out_dim=start_depth, name='conv1', bn=False)
        L.conv(out_dim=start_depth * 2, name='conv2', bn=True)
        L.conv(out_dim=start_depth * 4, name='conv3', bn=True)
        L.conv(out_dim=start_depth * 8, name='conv4', bn=True)

        L.linear(out_dim=1,
                 layer_dict=layer_dict,
                 init_w=init_w,
                 wd=0,
                 bn=False,
                 is_training=is_training,
                 name='Linear')

        return layer_dict['cur_input']
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: modules.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import src.models.layers as L

def DCGAN_discriminator(inputs, init_w, is_training, layer_dict,
                        start_depth=64, wd=0, name='DCGAN_discriminator'):
    with tf.variable_scope(name):
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

def BEGAN_encoder(inputs, layer_dict, n_code=64, start_depth=64, nl=tf.nn.relu,
                  init_w=None, is_training=True, bn=False, wd=0, name='encoder'):
    with tf.variable_scope(name):
        layer_dict['cur_input'] = inputs
        filter_size = 3

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.conv], 
                        filter_size=filter_size, layer_dict=layer_dict,
                        nl=tf.nn.elu, add_summary=False,
                        init_w=init_w, wd=wd, is_training=is_training, bn=bn):

            L.conv(out_dim=start_depth, name='conv0', stride=1)

            L.conv(out_dim=start_depth, name='conv1', stride=1)
            L.conv(out_dim=start_depth * 2, name='conv2', stride=2)

            L.conv(out_dim=start_depth * 2, name='conv3', stride=1)
            L.conv(out_dim=start_depth * 3, name='conv4', stride=2)

            L.conv(out_dim=start_depth * 3, name='conv5', stride=1)
            L.conv(out_dim=start_depth * 3, name='conv6', stride=2)

            L.linear(out_dim=n_code,
                     layer_dict=layer_dict,
                     init_w=init_w,
                     wd=wd,
                     nl=nl,
                     bn=bn,
                     is_training=is_training,
                     name='Linear')
            return layer_dict['cur_input']

def BEGAN_decoder(inputs, layer_dict,
                  start_size=8, n_feature=64, n_channle=3,
                  init_w=None, is_training=True, bn=False, wd=0,
                  name='BEGAN_decoder'):

    with tf.variable_scope(name):
        layer_dict['cur_input'] = inputs
        filter_size = 3

        # print(layer_dict['cur_input'])

        h0 = L.linear(out_dim=start_size*start_size*n_feature,
                      layer_dict=layer_dict,
                      init_w=init_w, wd=wd,
                      bn=bn, is_training=is_training,
                      # nl=tf.nn.relu
                      name='Linear')
        h0 = tf.reshape(h0, [-1, start_size, start_size, n_feature])
        layer_dict['cur_input'] = h0

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.conv, L.transpose_conv], 
                        filter_size=filter_size, layer_dict=layer_dict,
                        nl=tf.nn.relu,
                        init_w=init_w, wd=wd, is_training=is_training, bn=bn):

            L.conv(out_dim=n_feature, stride=1, name='conv1')
            L.transpose_conv(out_dim=n_feature, stride=2, name='dconv2')

            L.conv(out_dim=n_feature, stride=1, name='conv3')
            L.transpose_conv(out_dim=n_feature, stride=2, name='dconv4') 

            L.conv(out_dim=n_feature, stride=1, name='conv5')
            # L.conv(out_dim=n_feature, stride=1, name='conv6')
            L.transpose_conv(out_dim=n_feature, stride=2, name='dconv6') 

            L.conv(out_dim=n_channle, stride=1, name='decoder_out',
                   nl=tf.tanh, bn=False) 

            return layer_dict['cur_input']

def LSGAN_generator(inputs, layer_dict,
                    im_size, n_channle=3,
                    init_w=None, keep_prob=1., wd=0,
                    is_training=True, bn=False, name='LSGAN_generator'):
    with tf.variable_scope(name):
        layer_dict['cur_input'] = inputs
        b_size = tf.shape(inputs)[0]

        d_height_2, d_width_2 = L.deconv_size(im_size[0], im_size[1])
        d_height_4, d_width_4 = L.deconv_size(d_height_2, d_width_2)
        d_height_8, d_width_8 = L.deconv_size(d_height_4, d_width_4)
        d_height_16, d_width_16 = L.deconv_size(d_height_8, d_width_8)

        L.linear(out_dim=d_height_16 * d_width_16 * 256,
                 layer_dict=layer_dict,
                 init_w=init_w,
                 wd=wd,
                 bn=bn,
                 is_training=is_training,
                 name='Linear',
                 nl=tf.nn.relu)
        layer_dict['cur_input'] = tf.reshape(
            layer_dict['cur_input'],
            [-1, d_height_16, d_width_16, 256])

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.transpose_conv], 
                        filter_size=3, layer_dict=layer_dict,
                        init_w=init_w, wd=wd, is_training=is_training):
            output_shape = [b_size, d_height_8, d_width_8, 256]
            L.transpose_conv(out_shape=output_shape, stride=2,
                             bn=bn, nl=tf.nn.relu, name='dconv1')
            L.transpose_conv(out_shape=output_shape, stride=1,
                             bn=bn, nl=tf.nn.relu, name='dconv2')
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)

            output_shape = [b_size, d_height_4, d_width_4, 256]
            L.transpose_conv(out_shape=output_shape, stride=2,
                             bn=bn, nl=tf.nn.relu, name='dconv3')
            L.transpose_conv(out_shape=output_shape, stride=1,
                             bn=bn, nl=tf.nn.relu, name='dconv4')
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)

            output_shape = [b_size, d_height_2, d_width_2, 128]
            L.transpose_conv(out_shape=output_shape, stride=2,
                             bn=bn, nl=tf.nn.relu, name='dconv5')

            output_shape = [b_size, im_size[0], im_size[1], 64]
            L.transpose_conv(out_shape=output_shape, stride=2,
                             bn=bn, nl=tf.nn.relu, name='dconv6')
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)

            output_shape = [b_size, im_size[0], im_size[1], n_channle]
            L.transpose_conv(out_shape=output_shape, stride=1,
                             bn=False, nl=tf.tanh, name='dconv7')

            return layer_dict['cur_input']


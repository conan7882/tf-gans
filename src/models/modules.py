#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: modules.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import tensorflow_probability as tfp
import src.models.layers as L


def ACGAN_generator(inputs, init_w, is_training, layer_dict,
                    im_h, im_w, n_channels,
                    final_dim=96, filter_size=5, wd=0, keep_prob=1.0,
                    name='ACGAN_generator'):
    """ ACGAN generator

    Args:
        inputs (tensor): input tensor in batch
        init_w: initializer for weights
        is_training (bool): whether for training or not
        layer_dict (dictionary): dictionary of model
        im_h, im_w, n_channel (int): dimemtion of generate image
        final_dim (int): number of features of the last conv layer
        filter_size (int): filter size of convolutional layers
        wd: weight decay weight
        keep_prob (float): keep probablity for dropout
        name (str)

    Return:
        tensor of discriminator output 
    """

    with tf.variable_scope(name):
        layer_dict['cur_input'] = inputs
        # final_dim = 64
        # filter_size = 5
        b_size = tf.shape(inputs)[0]

        d_height_2, d_width_2 = L.deconv_size(im_h, im_w)
        d_height_4, d_width_4 = L.deconv_size(d_height_2, d_width_2)
        d_height_8, d_width_8 = L.deconv_size(d_height_4, d_width_4)
        # d_height_16, d_width_16 = L.deconv_size(d_height_8, d_width_8)

        L.linear(out_dim=d_height_8 * d_height_8 * final_dim * 4,
                 layer_dict=layer_dict,
                 init_w=init_w,
                 wd=wd,
                 bn=False,
                 is_training=is_training,
                 name='Linear',
                 nl=tf.nn.relu)
        layer_dict['cur_input'] = tf.reshape(
            layer_dict['cur_input'],
            [-1, d_height_8, d_height_8, final_dim * 4])

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.transpose_conv], 
                       filter_size=filter_size, layer_dict=layer_dict,
                       init_w=init_w, wd=wd, is_training=is_training):

            output_shape = [b_size, d_height_4, d_width_4, final_dim * 2]
            L.transpose_conv(out_dim=final_dim * 2, out_shape=output_shape,
                             bn=True, nl=tf.nn.relu, name='dconv2')
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)

            output_shape = [b_size, d_height_2, d_width_2, final_dim]
            L.transpose_conv(out_dim=final_dim, out_shape=output_shape,
                             bn=True, nl=tf.nn.relu, name='dconv3')
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)

            output_shape = [b_size, im_h, im_w, n_channels]
            L.transpose_conv(out_dim=n_channels, out_shape=output_shape,
                             bn=False, nl=tf.tanh, name='dconv4')

            return layer_dict['cur_input']

def ACGAN_discriminator(inputs, init_w, is_training, layer_dict, out_dim=11,
                        start_depth=16, wd=0, keep_prob=0.5,
                        name='ACGAN_discriminator'):
    """ ACGAN discriminator

    Args:
        inputs (tensor): input tensor in batch
        init_w: initializer for weights
        is_training (bool): whether for training or not
        layer_dict (dictionary): dictionary of model
        start_depth (int): number of features of the first conv layer
        wd: weight decay weight
        name (str)

    Return:
        tensor of discriminator output 
    """
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
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)
            L.conv(out_dim=start_depth * 2, name='conv2', bn=True)
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)
            L.conv(out_dim=start_depth * 4, name='conv3', bn=True)
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)
            L.conv(out_dim=start_depth * 8, name='conv4', bn=True)
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)
            L.conv(out_dim=start_depth * 16, name='conv4', bn=True)
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)
            L.conv(out_dim=start_depth * 32, name='conv4', bn=True)
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)
            layer_dict['{}_conv'.format(name)] = layer_dict['cur_input']

            L.linear(out_dim=out_dim,
                     layer_dict=layer_dict,
                     init_w=init_w,
                     wd=0,
                     bn=False,
                     is_training=is_training,
                     name='Linear')

            return layer_dict['cur_input']

def InfoGAN_MNIST_generator(
        inputs, init_w, is_training, layer_dict,
        im_h, im_w, n_channels,
        final_dim=64, filter_size=4, wd=0, keep_prob=1.0,
        name='InfoGAN_MNIST_generator'):

    """ InfoGAN generator for MNIST

    Args:
        inputs (tensor): input tensor in batch
        init_w: initializer for weights
        is_training (bool): whether for training or not
        layer_dict (dictionary): dictionary of model
        im_h, im_w, n_channel (int): dimemtion of generate image
        final_dim (int): number of features of the last conv layer
        filter_size (int): filter size of convolutional layers
        wd: weight decay weight
        keep_prob (float): keep probablity for dropout
        name (str)

    Return:
        tensor of discriminator output 
    """

    with tf.variable_scope(name):
        layer_dict['cur_input'] = inputs
        b_size = tf.shape(inputs)[0]

        d_height_2, d_width_2 = L.deconv_size(im_h, im_w)
        d_height_4, d_width_4 = L.deconv_size(d_height_2, d_width_2)
        d_height_8, d_width_8 = L.deconv_size(d_height_4, d_width_4)
        d_height_16, d_width_16 = L.deconv_size(d_height_8, d_width_8)

        L.linear(out_dim=1024,
                 layer_dict=layer_dict,
                 init_w=init_w,
                 wd=wd,
                 bn=True,
                 is_training=is_training,
                 name='fc1',
                 nl=tf.nn.relu)

        L.linear(out_dim=d_height_8 * d_height_8 * final_dim * 4,
                 layer_dict=layer_dict,
                 init_w=init_w,
                 wd=wd,
                 bn=True,
                 is_training=is_training,
                 name='fc2',
                 nl=tf.nn.relu)

        layer_dict['cur_input'] = tf.reshape(
            layer_dict['cur_input'],
            [-1, d_height_8, d_height_8, final_dim * 4])

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.transpose_conv], 
                       filter_size=filter_size, layer_dict=layer_dict,
                       init_w=init_w, wd=wd, is_training=is_training):

            # output_shape = [b_size, d_height_8, d_height_8, final_dim * 4]
            # L.transpose_conv(out_dim=final_dim * 4, out_shape=output_shape,
            #                  bn=True, nl=tf.nn.relu, name='dconv3')
            # L.drop_out(layer_dict, is_training, keep_prob=keep_prob)

            output_shape = [b_size, d_height_4, d_width_4, final_dim * 2]
            L.transpose_conv(out_dim=final_dim * 2, out_shape=output_shape,
                             bn=True, nl=tf.nn.relu, name='dconv1')
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)

            output_shape = [b_size, d_height_2, d_width_2, final_dim]
            L.transpose_conv(out_dim=final_dim, out_shape=output_shape,
                             bn=True, nl=tf.nn.relu, name='dconv2')
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)

            output_shape = [b_size, im_h, im_w, n_channels]
            L.transpose_conv(out_dim=n_channels, out_shape=output_shape,
                             bn=False, nl=tf.tanh, name='dconv_out')

            return layer_dict['cur_input']

def DCGAN_generator(inputs, init_w, is_training, layer_dict,
                    im_h, im_w, n_channels,
                    final_dim=64, filter_size=5, wd=0, keep_prob=1.0,
                    name='DCGAN_generator'):
    """ DCGAN generator

    Args:
        inputs (tensor): input tensor in batch
        init_w: initializer for weights
        is_training (bool): whether for training or not
        layer_dict (dictionary): dictionary of model
        im_h, im_w, n_channel (int): dimemtion of generate image
        final_dim (int): number of features of the last conv layer
        filter_size (int): filter size of convolutional layers
        wd: weight decay weight
        keep_prob (float): keep probablity for dropout
        name (str)

    Return:
        tensor of discriminator output 
    """

    with tf.variable_scope(name):
        layer_dict['cur_input'] = inputs
        # final_dim = 64
        # filter_size = 5
        b_size = tf.shape(inputs)[0]

        d_height_2, d_width_2 = L.deconv_size(im_h, im_w)
        d_height_4, d_width_4 = L.deconv_size(d_height_2, d_width_2)
        d_height_8, d_width_8 = L.deconv_size(d_height_4, d_width_4)
        d_height_16, d_width_16 = L.deconv_size(d_height_8, d_width_8)

        L.linear(out_dim=d_height_16 * d_width_16 * final_dim * 8,
                 layer_dict=layer_dict,
                 init_w=init_w,
                 wd=wd,
                 bn=True,
                 is_training=is_training,
                 name='Linear',
                 nl=tf.nn.relu)
        layer_dict['cur_input'] = tf.reshape(
            layer_dict['cur_input'],
            [-1, d_height_16, d_width_16, final_dim * 8])

        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.transpose_conv], 
                       filter_size=filter_size, layer_dict=layer_dict,
                       init_w=init_w, wd=wd, is_training=is_training):
            output_shape = [b_size, d_height_8, d_width_8, final_dim * 4]
            L.transpose_conv(out_dim=final_dim * 4, out_shape=output_shape,
                             bn=True, nl=tf.nn.relu, name='dconv1')
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)

            output_shape = [b_size, d_height_4, d_width_4, final_dim * 2]
            L.transpose_conv(out_dim=final_dim * 2, out_shape=output_shape,
                             bn=True, nl=tf.nn.relu, name='dconv2')
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)

            output_shape = [b_size, d_height_2, d_width_2, final_dim]
            L.transpose_conv(out_dim=final_dim, out_shape=output_shape,
                             bn=True, nl=tf.nn.relu, name='dconv3')
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)

            output_shape = [b_size, im_h, im_w, n_channels]
            L.transpose_conv(out_dim=n_channels, out_shape=output_shape,
                             bn=False, nl=tf.tanh, name='dconv4')

            return layer_dict['cur_input']

def DCGAN_discriminator(inputs, init_w, is_training, layer_dict, out_dim=1,
                        start_depth=64, wd=0, name='DCGAN_discriminator'):
    """ DCGAN discriminator

    Args:
        inputs (tensor): input tensor in batch
        init_w: initializer for weights
        is_training (bool): whether for training or not
        layer_dict (dictionary): dictionary of model
        start_depth (int): number of features of the first conv layer
        wd: weight decay weight
        name (str)

    Return:
        tensor of discriminator output 
    """
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
            layer_dict['{}_conv'.format(name)] = layer_dict['cur_input']

            L.linear(out_dim=out_dim,
                     layer_dict=layer_dict,
                     init_w=init_w,
                     wd=0,
                     bn=False,
                     is_training=is_training,
                     name='Linear')

            return layer_dict['cur_input']

def BEGAN_encoder(inputs, layer_dict, n_code=64, start_depth=64, nl=tf.nn.relu,
                  init_w=None, is_training=True, bn=False, wd=0, name='encoder'):
    """ BEGAN encoder

    Args:
        inputs (tensor): input tensor in batch
        layer_dict (dictionary): dictionary of model
        n_code (int): dimension of code
        start_depth (int): number of features of the first conv layer
        nl: nonlinearity of output
        init_w: initializer for weights
        is_training (bool): whether for training or not
        bn (bool): whether apply batch normalization or not
        wd: weight decay weight
        name (str)

    Return:
        tensor of encode code 
    """
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
            L.conv(out_dim=start_depth * 4, name='conv6', stride=2)

            L.conv(out_dim=start_depth * 4, name='conv7', stride=1)
            L.conv(out_dim=start_depth * 4, name='conv8', stride=2)

            L.conv(out_dim=start_depth * 4, name='conv9', stride=1)
            L.conv(out_dim=start_depth * 4, name='conv10', stride=1)

            L.linear(out_dim=n_code,
                     layer_dict=layer_dict,
                     init_w=init_w,
                     wd=wd,
                     nl=nl,
                     bn=bn,
                     is_training=is_training,
                     name='Linear')
            return layer_dict['cur_input']

def BEGAN_NN_decoder(inputs, layer_dict,
                  start_size=8, n_feature=64, n_channle=3,
                  init_w=None, is_training=True, bn=False, wd=0,
                  name='BEGAN_decoder'):
    """ BEGAN decoder/generator upsampling using nearest neighborhood

    Args:
        inputs (tensor): input tensor in batch
        layer_dict (dictionary): dictionary of model
        start_size (int): side size of the first 2d tensor for conv
        n_feature (int): number of features of the first conv layer
        n_channle (int): number of channels of output image
        init_w: initializer for weights
        is_training (bool): whether for training or not
        bn (bool): whether apply batch normalization or not
        wd: weight decay weight
        name (str)

    Return:
        tensor of decode image 
    """

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
            L.conv(out_dim=n_feature, stride=1, name='conv2')
            L.NN_upsampling(layer_dict=layer_dict, factor=2, name='up1')

            L.conv(out_dim=2*n_feature, stride=1, name='conv3')
            L.conv(out_dim=n_feature, stride=1, name='conv4')
            # L.transpose_conv(out_dim=n_feature, stride=2, name='dconv4')
            L.NN_upsampling(layer_dict=layer_dict, factor=2, name='up2') 

            L.conv(out_dim=2*n_feature, stride=1, name='conv5')
            L.conv(out_dim=n_feature, stride=1, name='conv6')
            # L.transpose_conv(out_dim=n_feature, stride=2, name='dconv6')
            L.NN_upsampling(layer_dict=layer_dict, factor=2, name='up3') 

            L.conv(out_dim=2*n_feature, stride=1, name='conv7')
            L.conv(out_dim=n_feature, stride=1, name='conv8')

            L.NN_upsampling(layer_dict=layer_dict, factor=2, name='up4') 

            L.conv(out_dim=2*n_feature, stride=1, name='conv9')
            L.conv(out_dim=n_feature, stride=1, name='conv10')

            L.conv(out_dim=n_channle, stride=1, name='decoder_out',
                   nl=tf.tanh, bn=False) 

            return layer_dict['cur_input']

def BEGAN_decoder(inputs, layer_dict,
                  start_size=8, n_feature=64, n_channle=3,
                  init_w=None, is_training=True, bn=False, wd=0,
                  name='BEGAN_decoder'):
    """ BEGAN decoder also used as generator

    Args:
        inputs (tensor): input tensor in batch
        layer_dict (dictionary): dictionary of model
        start_size (int): side size of the first 2d tensor for conv
        n_feature (int): number of features of the first conv layer
        n_channle (int): number of channels of output image
        init_w: initializer for weights
        is_training (bool): whether for training or not
        bn (bool): whether apply batch normalization or not
        wd: weight decay weight
        name (str)

    Return:
        tensor of decode image 
    """

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
            L.transpose_conv(out_dim=n_feature, stride=2, name='dconv6') 

            # L.conv(out_dim=n_feature, stride=1, name='conv7')
            # L.transpose_conv(out_dim=n_feature, stride=2, name='dconv8') 

            L.conv(out_dim=n_channle, stride=1, name='decoder_out',
                   nl=tf.tanh, bn=False) 

            return layer_dict['cur_input']

def LSGAN_generator(inputs, layer_dict,
                    im_size, n_channle=3,
                    init_w=None, keep_prob=1., wd=0,
                    is_training=True, bn=False, name='LSGAN_generator'):
    """ LSGAN generator

    Args:
        inputs (tensor): input tensor in batch
        layer_dict (dictionary): dictionary of model
        im_size (int): output image size
        n_channle (int): number of channels of output image
        init_w: initializer for weights
        keep_prob (float): keep probability for dropout
        wd: weight decay weight
        is_training (bool): whether for training or not
        bn (bool): whether apply batch normalization or not
        name (str)

    Return:
        tensor of generate image 
    """
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

def categorical_distribution_layer(inputs, layer_dict, n_class,
                                   init_w, wd, is_training,
                                   name='categorical_distribution_layer'):
    """ Estimate a categorical distribution

    Args:
        inputs (tensor): input tensor in batch
        layer_dict (dictionary): dictionary of model
        n_class (int): number of classes
        init_w: initializer for weights
        wd: weight decay weight
        is_training (bool): whether for training or not
        name (str)

    Return:
        
    """
    with tf.variable_scope(name):
        L.linear(
            out_dim=128,
            layer_dict=layer_dict,
            inputs=inputs,
            init_w=init_w,
            nl=L.leaky_relu,
            # init_b=tf.zeros_initializer(),
            wd=wd,
            bn=True,
            is_training=is_training,
            name='fc1',
            add_summary=False)
        out = L.linear(
            out_dim=n_class,
            layer_dict=layer_dict,
            # inputs=inputs,
            init_w=init_w,
            # init_b=tf.zeros_initializer(),
            wd=wd,
            bn=False,
            is_training=is_training,
            name='fc2',
            add_summary=False)

        return out #[bsize, n_class]

def diagonal_Gaussian_layer(inputs, layer_dict, n_dim,
                            init_w, wd, is_training,
                            name='diagonal_Gaussian_layer'):
    """ Estimate a diagonal Gaussian distribution

    Args:
        inputs (tensor): input tensor in batch
        layer_dict (dictionary): dictionary of model
        n_dim (int): dimensionality of Gaussian
        init_w: initializer for weights
        wd: weight decay weight
        is_training (bool): whether for training or not
        name (str)

    Return:
        mean and covariance of Gaussian 
    """
    with tf.variable_scope(name):
        out_dim = n_dim * 2
        out_params = L.linear(
            out_dim=out_dim,
            layer_dict=layer_dict,
            inputs=inputs,
            init_w=init_w,
            # init_b=tf.zeros_initializer(),
            wd=wd,
            bn=False,
            is_training=is_training,
            name='Linear',
            add_summary=False)
        mu = out_params[:, :n_dim]
        sigma = L.softplus(out_params[:, n_dim:])

        return mu, sigma

def sample_diagonal_Gaussian_reparameterization_trick(mean, sigma, n_dim, b_size,
                                                      name='sample_diagonal_Gaussian_reparameterization_trick'):
    """ Batch sample from a diagonal Gaussian distribution using reparameterization trick

    Args:
        mean (float): mean of the diagonal Gaussian distribution in batch
        sigma (float): variance of the diagonal Gaussion in batch
        n_dim (int): dimensionality of Gaussian
        b_size (int): batch size
        name (str)

    Return:
        samples drawn from the diagonal Gaussian distribution [bsize n_dim]
    """
    with tf.name_scope(name):
        # mean_list = [0.0 for i in range(0, n_code)]
        # std_list = [1.0 for i in range(0, n_code)]

        mean_list = tf.zeros_like(mean)
        std_list = tf.ones_like(mean)
        mvn = tfp.distributions.MultivariateNormalDiag(
            loc=mean_list,
            scale_diag=std_list)
        samples = mvn.sample(sample_shape=(b_size,), seed=None, name='standard_gaussian_sample')
        samples = mean +  tf.multiply(sigma, samples)

        return samples

def evaluate_log_diagonal_Gaussian_pdf(mean, sigma, samples, name='evaluate_diagonal_Gaussian_pdf'):
    """ Batch evaluate diagonal Gaussian distribution

    Args:
        mean (float): mean of the diagonal Gaussian distribution in batch
        sigma (float): variance of the diagonal Gaussion in batch
        samples: batch samples drawn from the diagonal Gaussian distribution

    Return:
        batch tensor [bsize n_dim]
    """
    with tf.name_scope(name):
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=sigma)
        log_prob = dist.log_prob(samples)
        # return tf.where(tf.is_nan(log_prob),
        #                 tf.zeros(tf.shape(log_prob)),
        #                 log_prob)
        return dist.log_prob(samples)

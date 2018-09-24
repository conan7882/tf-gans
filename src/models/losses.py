#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: losses.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf


def generator_cross_entropy_loss(d_fake, name='generator_loss'):
    """ cross entropy loss for generator

    Args:
        d_fake (tensor): discrimator output of fake data
        name (str)

    Return:
        cross entropy loss of generator
    """
    with tf.name_scope(name):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(d_fake),
            logits=d_fake,
            name='cross_entropy_fake')
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

def discriminator_cross_entropy_loss(d_fake, d_real, name='discriminator_loss'):
    """ cross entropy loss for discriminator

    Args:
        d_fake (tensor): discrimator output of fake data
        d_real (tensor): discrimator output of real data
        name (str)

    Return:
        cross entropy loss of discriminator
    """
    with tf.name_scope(name):
        loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(d_real),
            logits=d_real,
            name='cross_entropy_real')
        loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(d_fake),
            logits=d_fake,
            name='cross_entropy_fake')
        d_loss = tf.reduce_mean(loss_real) + tf.reduce_mean(loss_fake)
        return d_loss

def generator_least_square_loss(d_fake_logits, c=1., name='generator_loss'):
    """ least square loss for generator

    Args:
        d_fake_logits (tensor): discrimator output of fake data
        c (float): the value that generator wants discriminator
            to believe for fake data
        name (str)

    Return:
        least square loss of generator
    """
    with tf.name_scope(name):
        # d_fake = tf.sigmoid(d_fake_logits)
        d_fake = d_fake_logits
        loss_fake = tf.reduce_mean((d_fake - c) ** 2, name='least_square_fake')
        return 0.5 * loss_fake

def discriminator_least_square_loss(d_fake_logits, d_real_logits,
                                    a=0., b=1., name='discriminator_loss'):
    """ least square loss for discriminator

    Args:
        d_fake_logits (tensor): discrimator output of fake data
        d_real_logits (tensor): discrimator output of real data
        a (float): labels for fake data
        b (float): labels for real data
        name (str)

    Return:
        least square loss of discriminator
    """
    with tf.name_scope(name):
        # d_fake = tf.sigmoid(d_fake_logits)
        # d_real = tf.sigmoid(d_real_logits)
        d_fake = d_fake_logits
        d_real = d_real_logits
        loss_fake = tf.reduce_mean((d_fake - a) ** 2, name='least_square_fake')
        loss_real = tf.reduce_mean((d_real - b) ** 2, name='least_square_real')
        return 0.5 * (loss_fake + loss_real)

def l2_loss(x, y):
    """ l2 loss """
    with tf.name_scope('l2_loss'):
        # return tf.reduce_mean(tf.sqrt(tf.reduce_sum((x - y) ** 2)))
        return tf.reduce_mean((x - y) ** 2)

def l1_loss(x, y):
    """ l1 loss """
    with tf.name_scope('l1_loss'):
        return tf.reduce_mean(tf.abs(x - y))

def max_log_likelihood_loss(probability, name='max_log_likelihood_loss'):
    with tf.name_scope(name):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(probability),
            logits=probability,
            name='cross_entropy')
        cross_entropy = tf.reduce_sum(cross_entropy, axis=-1)
        return tf.reduce_mean(cross_entropy)


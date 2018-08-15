#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: losses.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf


def generator_cross_entropy_loss(d_fake, name='generator_loss'):
    with tf.name_scope(name):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(d_fake),
            logits=d_fake,
            name='cross_entropy_fake')
        cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy

def discriminator_cross_entropy_loss(d_fake, d_real, name='discrimator_loss'):
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
    with tf.name_scope(name):
        # d_fake = tf.sigmoid(d_fake_logits)
        d_fake = d_fake_logits
        loss_fake = tf.reduce_mean((d_fake - c) ** 2, name='least_square_fake')
        return 0.5 * loss_fake

def discriminator_least_square_loss(d_fake_logits, d_real_logits,
                                    a=0., b=1., name='discriminator_loss'):
    with tf.name_scope(name):
        # d_fake = tf.sigmoid(d_fake_logits)
        # d_real = tf.sigmoid(d_real_logits)
        d_fake = d_fake_logits
        d_real = d_real_logits
        loss_fake = tf.reduce_mean((d_fake - a) ** 2, name='least_square_fake')
        loss_real = tf.reduce_mean((d_real - b) ** 2, name='least_square_real')
        return 0.5 * (loss_fake + loss_real)

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
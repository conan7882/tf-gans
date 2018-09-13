#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: loader.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import sys
import platform
import scipy.misc
import numpy as np

sys.path.append('../')
from src.dataflow.mnist import MNISTData
from src.dataflow.celeba import CelebA


def load_mnist(batch_size, shuffle=True, n_use_label=None, n_use_sample=None,
               rescale_size=None):
    """ Function for load training data 

    If n_use_label or n_use_sample is not None, samples will be
    randomly picked to have a balanced number of examples

    Args:
        batch_size (int): batch size
        n_use_label (int): how many labels are used for training
        n_use_sample (int): how many samples are used for training

    Retuns:
        MNISTData dataflow
    """
    data_path = '/home/qge2/workspace/data/MNIST_data/'

    def preprocess_im(im):
        """ normalize input image to [-1., 1.] """
        if rescale_size is not None:
            im = np.squeeze(im, axis=-1)
            im = scipy.misc.imresize(im, [rescale_size, rescale_size])
            im = np.expand_dims(im, axis=-1)
        im = im / 255. * 2. - 1.

        return np.clip(im, -1., 1.)

    data = MNISTData('train',
                     data_dir=data_path,
                     shuffle=shuffle,
                     pf=preprocess_im,
                     n_use_label=n_use_label,
                     n_use_sample=n_use_sample,
                     batch_dict_name=['im', 'label'])
    data.setup(epoch_val=0, batch_size=batch_size)
    return data

def load_celeba(batch_size, rescale_size=64, shuffle=True):
    """ Load CelebA data

    Args:
        batch_size (int): batch size
        rescale_size (int): rescale image size
        shuffle (bool): whether shuffle data or not

    Retuns:
        CelebA dataflow
    """
    data_path = '/home/qge2/workspace/data/celebA/'
        
    def face_preprocess(im):
        offset_h = 50
        offset_w = 25
        crop_h = 128
        crop_w = 128
        im = im[offset_h: offset_h + crop_h,
                offset_w: offset_w + crop_w, :]
        im = scipy.misc.imresize(im, [rescale_size, rescale_size])
        im = im / 255. * 2. - 1.
        return np.clip(im, -1., 1.)

    data = CelebA(data_dir=data_path, shuffle=shuffle,
                  batch_dict_name=['im'], pf_list=[face_preprocess])
    data.setup(epoch_val=0, batch_size=batch_size)
    return data

if __name__ == '__main__':
    import matplotlib.pyplot as plt
        
    data = load_celeba(10)
    batch_data = data.next_batch_dict()

    cur_im = np.squeeze(batch_data['im'][0])
    cur_im = ((cur_im + 1) * 255 / 2)
    cur_im = cur_im.astype(np.uint8)

    plt.figure()
    plt.imshow(cur_im)
    plt.show()

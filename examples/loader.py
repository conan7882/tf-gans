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
        MNISTData

    """
    if platform.node() == 'Qians-MacBook-Pro.local':
        data_path = '/Users/gq/workspace/Dataset/MNIST_data/'
    elif platform.node() == 'arostitan':
        data_path = '/home/qge2/workspace/data/MNIST_data/'
    elif platform.node() == 'aros04':
        data_path = 'E://Dataset//MNIST//'
    else:
        raise ValueError('Data path does not setup on this platform!')

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
    if platform.node() == 'aros04':
        data_path = 'E:/Dataset/celebA/Img/img_align_celeba_png.7z/img_align_celeba_png/'
    elif platform.node() == 'arostitan':
        data_path = '/home/qge2/workspace/data/celebA/'
    else:
        raise ValueError('Data path does not setup on this platform!')
        
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
    
    # im_path = os.path.join(datapath, '{:06d}.png'.format(5))
    # im = scipy.misc.imread(im_path, mode='RGB')
    # im = im[50:50+128,25:25+128, :]

    # plt.figure()
    # plt.imshow(im)
    # plt.show()

    
    data = load_celeba(10)
    batch_data = data.next_batch_dict()

    cur_im = np.squeeze(batch_data['im'][0])
    cur_im = ((cur_im + 1) * 255 / 2)
    cur_im = cur_im.astype(np.uint8)

    plt.figure()
    plt.imshow(cur_im)
    plt.show()


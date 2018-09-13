#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: celeba.py
# Author: Qian Ge <geqian1001@gmail.com>

import scipy.misc

from src.dataflow.base import DataFlow
import src.utils.utils as utils
from src.utils.dataflow import load_image, identity, fill_pf_list


class CelebA(DataFlow):
    """ dataflow for CelebA dataset """
    def __init__(self,
                 data_dir='',
                 shuffle=True,
                 batch_dict_name=None,
                 pf_list=None):
        """
        Args:
            data_dir (str): directory of data
            shuffle (bool): whether shuffle data or not
            batch_dict_name (str): key of face image when getting batch data
            pf_list: pre-process functions for face image
        """

        pf_list = fill_pf_list(
            pf_list, n_pf=1, fill_with_fnc=identity)

        def read_image(file_name):
            """ read color face image with pre-process function """
            face_image = load_image(file_name, read_channel=3,  pf=pf_list[0])
            return face_image

        super(CelebA, self).__init__(
            data_name_list=['.png'],
            data_dir=data_dir,
            shuffle=shuffle,
            batch_dict_name=batch_dict_name,
            load_fnc_list=[read_image],
            ) 



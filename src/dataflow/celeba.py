#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: celeba.py
# Author: Qian Ge <geqian1001@gmail.com>

import scipy.misc

from src.dataflow.base import DataFlow
import src.utils.utils as utils
from src.utils.dataflow import load_image, identity, fill_pf_list


class CelebA(DataFlow):
    def __init__(self,
                 data_dir='',
                 shuffle=True,
                 batch_dict_name=None,
                 pf_list=None):
        pf_list = fill_pf_list(
            pf_list, n_pf=1, fill_with_fnc=identity)

        def read_image(file_name):
            face_image = load_image(file_name, read_channel=3,  pf=pf_list[0])
            return face_image

        super(CelebA, self).__init__(
            data_name_list=['.png'],
            data_dir=data_dir,
            shuffle=shuffle,
            batch_dict_name=batch_dict_name,
            load_fnc_list=[read_image],
            ) 



#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: entropies.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np


def entropy_uniform(n_class):
    """ compute entropy of uniform distribution

    Args:
        n_class (int): P(X = xn) = 1/n_class
    """
    return np.log(n_class)

def entropy_isotropic_Gaussian(n_gaussian, sigma=1.0):
    """ compute entropy of isotropic Gaussians

    Args:
        n_gaussian (int): number of dimension
        sigma (float): variance of the Gaussian
    """
    if n_gaussian == 0 or sigma == 0:
        return 0
    return 0.5 * np.log(sigma ** n_gaussian) + n_gaussian / 2. * (1 + np.log(2. * np.pi))


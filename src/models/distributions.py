#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: distributions.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np


def random_vector(vec_shape, dist_type='gaussian'):
    if dist_type == 'uniform':
        return np.random.uniform(-1, 1, size=vec_shape)
    else:
        return np.random.normal(size=vec_shape)

def linear_2D_interpolate(side_size=20, interpolate_range=[-1, 1, -1, 1]):
    assert len(interpolate_range) == 4
    nx = side_size
    ny = side_size
    min_x = interpolate_range[0]
    max_x = interpolate_range[1]
    min_y = interpolate_range[2]
    max_y = interpolate_range[3]
    
    zs = np.rollaxis(np.mgrid[min_x: max_x: nx*1j, max_y:min_y: ny*1j], 0, 3)
    zs = zs.transpose(1, 0, 2)
    return np.reshape(zs, (side_size * side_size, 2))

def linear_interpolate(z1=None, z2=None, z_shape=None, n_samples=10):
    if z1 is None:
        z1 = np.random.uniform(-1, 1, size=z_shape)
    else:
        z1 = np.array(z1)

    if z2 is None:
        z2 = np.random.uniform(-1, 1, size=z_shape)
    else:
        z2 = np.array(z2)

    dz = (z2 - z1) / float(n_samples)
    z = np.array([z1 + i * dz for i in range(n_samples)], dtype=np.float32)
    return z

def great_circle_interpolate(z1=None, z2=None, z_shape=None, n_samples=10):
    if z1 is None:
        z1 = np.random.uniform(-1, 1, size=z_shape)
    else:
        z1 = np.array(z1)

    if z2 is None:
        z2 = np.random.uniform(-1, 1, size=z_shape)
    else:
        z2 = np.array(z2)

    val = 1. / float(n_samples)
    interp_list = []
    for cur_z1, cur_z2 in zip(z1, z2):
        interp_list.append(slerp(val, cur_z1, cur_z2))
    return interp_list

def slerp(val, low, high):
    """ Interpolation on the unit n-sphere 

        borrow from:
        https://github.com/soumith/dcgan.torch/issues/14
    """
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high

if __name__ == "__main__":
    z = linear_interpolate(z1=None, z2=None, z_shape=(1, 2), n_samples=10)
    print(z)


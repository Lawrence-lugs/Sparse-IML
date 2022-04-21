# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 18:47:39 2022

@author: Lawrence
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from __future__ import division, print_function
from builtins import input
import pickle


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = mpimg.imread();
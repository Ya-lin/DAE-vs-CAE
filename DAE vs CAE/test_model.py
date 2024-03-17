#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 11:14:07 2024

@author: klein
"""
import numpy as np
from matplotlib import pyplot as plt

def img2img_hat(noisy_imgs, imgs_hat):
    fig, axes = plt.subplots(nrows=2, ncols=10, 
                             sharex=True, sharey=True, 
                             figsize=(25, 4))
    for noisy_imgs, row in zip([noisy_imgs, imgs_hat], axes):
        for img, ax in zip(noisy_imgs, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
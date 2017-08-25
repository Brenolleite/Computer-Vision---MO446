import cv2
import copy as cp
import numpy as np
import masks as mask
import convolution as cv
import bi_interpolation as itp

import os
import math

def pyrDown(img, lvls):
    pyramid = []
    pyramid.append(img)

    for i in range(0, lvls):
        cv2.imwrite('../output/p1-2-2-down-{}.png'.format(i), img)
        img = pyrDownLvl(cp.copy(img))
        pyramid.append(img)

    return pyramid

def pyrDownLvl(img):
    b_img = cv.convolve(cp.copy(img), mask.g_3)
    i_height, i_width, i_channels = img.shape

    n_height = math.floor(i_height / 2)
    n_width = math.floor(i_width / 2)

    n_level = np.zeros((n_height, n_width, 3), np.uint8)

    for i in range(0, n_height):
        for j in range(0, n_width):
            n_level.itemset((i, j, 0), b_img.item(i * 2, j * 2, 0))
            n_level.itemset((i, j, 1), b_img.item(i * 2, j * 2, 1))
            n_level.itemset((i, j, 2), b_img.item(i * 2, j * 2, 2))

    return n_level

def pyrUp(img, lvls):
    pyramid = []
    pyramid.append(img)

    for i in range(0, lvls):
        cv2.imwrite('../output/p1-2-2-up-{}.png'.format(i), img)
        img = pyrUpLvl(cp.copy(img))
        pyramid.append(img)

    return pyramid

def pyrUpLvl(img):
    i_height, i_width, i_channels = img.shape

    n_height = math.floor(i_height * 2)
    n_width = math.floor(i_width * 2)

    n_level = np.zeros((n_height, n_width, 3), np.uint8)

    for i in range(0, i_height):
        for j in range(0, i_width):
            n_level.itemset((i * 2, j * 2, 0), img.item(i, j, 0))
            n_level.itemset((i * 2, j * 2, 1), img.item(i, j, 1))
            n_level.itemset((i * 2, j * 2, 2), img.item(i, j, 2))

    n_level = itp.interpolate(cp.copy(n_level))

    return n_level

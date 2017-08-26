import cv2
import copy as cp
import numpy as np
import masks as mask
import convolution as cv
import bi_interpolation as itp
import gaussian_pyramid as gPyr

import os
import math

def pyrDown(input, lvls):
    pyramid = []
    g_pyramid = gPyr.pyrDown(cp.copy(input), lvls)

    for i in range(lvls - 1):
        current_img = g_pyramid[i]
        output = pyrDownLvl(current_img, g_pyramid[i + 1])
        pyramid.append(output)

        cv2.imwrite('../output/p1-2-3-down-{}.png'.format(i), output)

    i += 1
    pyramid.append(g_pyramid[i])

    cv2.imwrite('../output/p1-2-3-down-{}.png'.format(i), g_pyramid[i])

    return pyramid

def pyrDownLvl(input, g_image):
    current_img = input
    b_img = gPyr.pyrUpLvl(g_image)

    if b_img.shape[0] < current_img.shape[0]:
        current_img = np.delete(current_img,(-1),axis=0)
    if b_img.shape[1] < current_img.shape[1]:
        current_img = np.delete(current_img,(-1),axis=1)

    return (current_img - b_img)

def pyrUp(input, lvl):
    pyramid = []
    j = len(input) - 1

    c_img = input[j]
    pyramid.append(c_img)
    cv2.imwrite('../output/p1-2-3-up-0.png', c_img)

    for i in range(lvl - 1):
        if (j > 0):
            j -= 1

        output = pyrUpLvl(cp.copy(c_img), input[j])
        pyramid.append(output)
        c_img = output

        cv2.imwrite('../output/p1-2-3-up-{}.png'.format(i + 1), output)

    return pyramid

def pyrUpLvl(input, h_pass):
    current_img = input
    extended = gPyr.pyrUpLvl(cp.copy(current_img))

    while extended.shape[0] < h_pass.shape[0]:
        h_pass = np.delete(h_pass, (-1), axis = 0)
    while extended.shape[1] < h_pass.shape[1]:
        h_pass = np.delete(h_pass, (-1), axis = 1)

    return (extended + h_pass)

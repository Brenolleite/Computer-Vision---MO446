import cv2
import copy as cp
import numpy as np
import masks as mask
import convolution as cv
import bi_interpolation as itp
import gaussian_pyramid as gPyr
import place_pyramid as pPyr

import os
import math

def blend(img1, img2, mask):
    lvl = 5
    g_p1 = gPyr.pyrDown(img1, 5)
    print("gp1: ", g_p1[4].shape)
    g_p2 = gPyr.pyrDown(img2, 5)
    print("gp2: ", g_p2[4].shape)
    g_pm = gPyr.pyrDown(mask, 5)
    print("gpm: ", g_pm[4].shape)

    l_p1 = pPyr.pyrDown(img1, 5)
    print("lp1: ", l_p1[4].shape)
    l_p2 = pPyr.pyrDown(img2, 5)
    print("lp2: ", l_p2[4].shape)

    t1 = l_p1[4] * (g_pm[4] / 255)
    cv2.imwrite('../output/t1.png', t1)

    t2 = l_p2[4] * (1 - (g_pm[4] / 255))
    cv2.imwrite('../output/t2.png', t2)

    d3 = t1 + t2
    cv2.imwrite('../output/t3.png', d3)

    for i in range(lvl - 1, 0, -1):
        print("Lvl: ", i)
        print("SHAPE 0: ", l_p1[i - 1].shape[0])
        print("SHAPE 0: ", g_pm[i - 1].shape[0])
        while l_p1[i - 1].shape[0] < g_pm[i - 1].shape[0]:
            g_pm[i - 1] = np.delete(g_pm[i - 1], (-1), axis = 0)
        while l_p1[i - 1].shape[1] < g_pm[i - 1].shape[1]:
            g_pm[i - 1] = np.delete(g_pm[i - 1], (-1), axis = 1)

        t1 = l_p1[i - 1] * (g_pm[i - 1] / 255)


        t2 = l_p2[i - 1] * (1 - (g_pm[i - 1] / 255))

        t3 = t1 + t2

        cv2.imwrite('../output/highpass-{}.png'.format(i-1), t3)

        d3 = pPyr.pyrUpLvl(d3, t3)

        cv2.imwrite('../output/d3-{}.png'.format(i-1), d3)

    output = np.zeros(img1.shape)
    return output

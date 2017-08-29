import cv2
import numpy as np
import gaussian_pyramid as gPyr
import place_pyramid as pPyr
import math

def blend(img1, img2, mask, lvl):
    # Image number controller
    z = 0

    g_pm = gPyr.gaussianPyramid(mask, lvl)

    l_p1 = pPyr.placePyramid(img1, lvl)
    l_p2 = pPyr.placePyramid(img2, lvl)

    i1_mask = l_p1[lvl - 1] * (g_pm[lvl - 1] / 255)
    # cv2.imwrite('output/report/p1-2-4-{}.png'.format(z), i1_mask)
    z += 1

    i2_mask = l_p2[lvl - 1] * (1 - (g_pm[lvl - 1] / 255))
    # cv2.imwrite('output/report/p1-2-4-{}.png'.format(z), i2_mask)
    z += 1

    blend_img = i1_mask + i2_mask
    # cv2.imwrite('output/report/p1-2-4-{}.png'.format(z), blend_img)
    z += 1

    for i in range(lvl - 1, 0, -1):
        while l_p1[i - 1].shape[0] < g_pm[i - 1].shape[0]:
            g_pm[i - 1] = np.delete(g_pm[i - 1], (-1), axis = 0)
        while l_p1[i - 1].shape[1] < g_pm[i - 1].shape[1]:
            g_pm[i - 1] = np.delete(g_pm[i - 1], (-1), axis = 1)

        i1_mask = l_p1[i - 1] * (g_pm[i - 1] / 255)


        i2_mask = l_p2[i - 1] * (1 - (g_pm[i - 1] / 255))

        high_pass = i1_mask + i2_mask

        blend_img = pPyr.pyrExpand(blend_img, high_pass)

    return blend_img

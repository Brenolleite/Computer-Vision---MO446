import cv2
import numpy as np
import gaussian_pyramid as gPyr
import place_pyramid as pPyr
import math

# Blends img1 and img2 using mask and a number of levels for all the pyramids in
# the algorithm
def blend(img1, img2, mask, lvl):
    # Image number controller
    z = 0

    # Generate the gaussian pyramids
    g_pm = gPyr.gaussianPyramid(mask, lvl)

    # Generate the laplacian pyramid, gaussian generated inside of it
    l_p1 = pPyr.placePyramid(img1, lvl)
    l_p2 = pPyr.placePyramid(img2, lvl)

    # Multiplication of img1 and the mask
    i1_mask = l_p1[lvl - 1] * (g_pm[lvl - 1] / 255)
    z += 1

    # Multiplication of img2 and the negative of mask
    i2_mask = l_p2[lvl - 1] * (1 - (g_pm[lvl - 1] / 255))
    # cv2.imwrite('output/report/p1-2-4-{}.png'.format(z), i2_mask)
    z += 1

    # Sum of img1 and img2 after masked
    blend_img = i1_mask + i2_mask
    z += 1

    # Reconstruction of the levels
    for i in range(lvl - 1, 0, -1):
        while l_p1[i - 1].shape[0] < g_pm[i - 1].shape[0]:
            g_pm[i - 1] = np.delete(g_pm[i - 1], (-1), axis = 0)
        while l_p1[i - 1].shape[1] < g_pm[i - 1].shape[1]:
            g_pm[i - 1] = np.delete(g_pm[i - 1], (-1), axis = 1)

        # Generation of the next multiplied img1 with mask
        i1_mask = l_p1[i - 1] * (g_pm[i - 1] / 255)

        # Generation of the next multiplied img2 with mask
        i2_mask = l_p2[i - 1] * (1 - (g_pm[i - 1] / 255))

        # Sum of the masked images
        high_pass = i1_mask + i2_mask

        # Expansion of the summed image to be used in the next level computation
        blend_img = pPyr.pyrExpand(blend_img, high_pass)

    return blend_img

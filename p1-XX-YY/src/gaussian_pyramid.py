import cv2
import copy as cp
import numpy as np
import masks as mask
import convolution as cv
import bi_interpolation as itp
import math

def gaussianPyramid(img, lvl):
    gPyramid = []
    gPyramid.append(img)

    aux = img

    for i in range(lvl):
        cv2.imwrite('../output/p1-2-2-{}.png'.format(i), aux)

        aux = pyrContract(aux)
        gPyramid.append(aux)

    return gPyramid

def pyrContract(current_img):
    blur_img = cv.convolve(cp.copy(current_img), mask.g_3)

    height = math.floor(current_img.shape[0] / 2)
    width = math.floor(current_img.shape[1] / 2)
    channel = current_img.shape[2]

    aux = np.zeros((height, width, channel), np.uint8)

    for i in range(height):
        for j in range(width):
            for k in range(channel):
                aux.itemset((i, j, k), blur_img.item(i * 2, j * 2, k))

    return aux

def pyrExpand(current_img):
    height, width, channel = current_img.shape

    aux = np.zeros((height * 2, width * 2, channel), np.uint8)

    for i in range(height):
        for j in range(width):
            for k in range(channel):
                aux.itemset((i * 2, j * 2, k), current_img.item(i, j, k))

    aux = itp.interpolate(aux)

    return aux

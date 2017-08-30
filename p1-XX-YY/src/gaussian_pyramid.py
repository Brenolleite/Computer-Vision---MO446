import cv2
import copy as cp
import numpy as np
import masks as mask
import convolution as cv
import bi_interpolation as itp
import math

# Caller to generate the gaussian pyramid
def gaussianPyramid(img, lvl):
    gPyramid = []
    gPyramid.append(img)
    lvl -= 1

    aux = img

    for i in range(lvl):
        aux = pyrContract(aux)
        gPyramid.append(aux)

    return gPyramid

# Function that generates a half sized image of the input by sampling
def pyrContract(current_img):
    # Convolves input with the gaussian mask
    blur_img = cv.convolve(cp.copy(current_img), mask.g_3)

    height = math.floor(current_img.shape[0] / 2)
    width = math.floor(current_img.shape[1] / 2)
    channel = current_img.shape[2]

    aux = np.zeros((height, width, channel), np.uint8)

    for i in range(height):
        for j in range(width):
            for k in range(channel):
                # Skip every other pixel of input
                aux.itemset((i, j, k), blur_img.item(i * 2, j * 2, k))

    return aux

# Function that generates a double sized image of the input by interpolating
def pyrExpand(current_img):
    height, width, channel = current_img.shape

    aux = np.zeros((height * 2, width * 2, channel), np.uint8)

    for i in range(height):
        for j in range(width):
            for k in range(channel):
                # Generates an image from input missing every other pixel
                aux.itemset((i * 2, j * 2, k), current_img.item(i, j, k))

    # Interpolate the missing pixels with their surroundings
    aux = itp.interpolate(aux)

    return aux

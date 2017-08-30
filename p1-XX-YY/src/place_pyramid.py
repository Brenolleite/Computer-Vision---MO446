import cv2
import numpy as np
import gaussian_pyramid as gPyr
import math

# Caller to generate the laplacian pyramid, the last level is added in here
def placePyramid(img, lvl):
    lPyramid = []
    gPyramid = gPyr.gaussianPyramid(img, lvl)

    for i in range(lvl - 1):
        low_pass = gPyr.pyrExpand(gPyramid[i + 1])
        current_img = gPyramid[i]

        aux = pyrContract(current_img, low_pass)

        lPyramid.append(aux)

    i += 1
    lPyramid.append(gPyramid[i])

    return lPyramid

# Function that generates a half sized image of the input, result is the sharp
# details of input
def pyrContract(current_img, low_pass):
    while low_pass.shape[0] < current_img.shape[0]:
        current_img = np.delete(current_img,(-1),axis=0)
    while low_pass.shape[1] < current_img.shape[1]:
        current_img = np.delete(current_img,(-1),axis=1)

    aux = current_img
    aux = aux.astype(np.int16)

    for i in range(aux.shape[0]):
        for j in range(aux.shape[1]):
            for k in range(aux.shape[2]):
                # Subtract every pixel from a blured version of it, result are
                # the sharps details
                pixel = aux.item(i, j, k) - low_pass.item(i, j, k)
                aux.itemset((i, j, k), pixel)

    return aux

# Function that generates a double sized image of the input, the result is the
# way towards the original image
def pyrExpand(current_img, high_pass):
    # Uses the gaussian expansion
    low_pass = gPyr.pyrExpand(current_img)

    while low_pass.shape[0] < high_pass.shape[0]:
        high_pass = np.delete(high_pass, (-1), axis = 0)
    while low_pass.shape[1] < high_pass.shape[1]:
        high_pass = np.delete(high_pass, (-1), axis = 1)

    # Sums the expanded image with the sharp details saved previously
    aux = low_pass + high_pass

    return aux

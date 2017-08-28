import cv2
import numpy as np
import gaussian_pyramid as gPyr
import math

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
                pixel = aux.item(i, j, k) - low_pass.item(i, j, k)
                aux.itemset((i, j, k), pixel)

    return aux

def pyrExpand(current_img, high_pass):
    low_pass = gPyr.pyrExpand(current_img)

    while low_pass.shape[0] < high_pass.shape[0]:
        high_pass = np.delete(high_pass, (-1), axis = 0)
    while low_pass.shape[1] < high_pass.shape[1]:
        high_pass = np.delete(high_pass, (-1), axis = 1)

    aux = low_pass + high_pass

    return aux

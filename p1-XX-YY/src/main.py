import cv2
import copy as cp
import numpy as np
import utils as ut

import masks as mask
import convolution as cv
import gaussian_pyramid as gPyr
import place_pyramid as pPyr
import blending as bl

# Python uses the BGR color scheme
input = cv2.imread('../input/messi5.png')

# 2.1
time = ut.time()
output = cv.convolve(cp.copy(input), mask.g_3)
print("Convolution time:" + time.elapsed())

#cv2.imwrite('../output/p1-1-0.png', output)

time = ut.time()

output = cv2.filter2D(cp.copy(input), -1, np.flip(np.flip(mask.g_3, 0), 1))

print("OpenCV Convolution time:" + time.elapsed())

# cv2.imwrite('../output/p1-1-1.png', output)

# 2.2
# gPyramidDown = gPyr.gauPyrDown(cp.copy(input), 5)
# gPyramidUp = gPyr.gauPyrUp(cp.copy(input), 3)

# 2.3
lPyramid = pPyr.placePyramid(input, 4)

# 2.4
img1 = cv2.imread('../input/img1.png')
img2 = cv2.imread('../input/img2.png')
b_mask = cv2.imread('../input/b_mask.png')

# output = bl.blend(img1, img2, b_mask)

# cv2.imwrite('../output/blended.png', output)

# Helpers

# OpenCV LaPlace Pyramid
def CVLaPlacePyramid(img, lvl):
    lPyramid = []
    gauPyramid = gPyr.pyrDown(img, lvl)

    for i in range(lvl - 1):
        low_pass = gPyr.pyrUpLvl(gauPyramid[i + 1])
        current_img = gauPyramid[i]

        aux = CVLaPlacePyramidContract(current_img, low_pass)

        cv2.imwrite('../output/OpenCV-LaPlace-Contract-{}.png'.format(i), aux)
        lPyramid.append(aux)

    i += 1
    cv2.imwrite('../output/OpenCV-LaPlace-Contract-{}.png'.format(i), gauPyramid[i])
    lPyramid.append(gauPyramid[i])

    return lPyramid

def CVLaPlacePyramidContract(current_img, low_pass):
    while low_pass.shape[0] < current_img.shape[0]:
        current_img = np.delete(current_img,(-1),axis=0)
    while low_pass.shape[1] < current_img.shape[1]:
        current_img = np.delete(current_img,(-1),axis=1)

    cv2.imwrite('../output/1.png', current_img)
    cv2.imwrite('../output/2.png', low_pass)

    aux = current_img
    aux = aux.astype(np.int16)

    for i in range(aux.shape[0]):
        for j in range(aux.shape[1]):
            for k in range(aux.shape[2]):
                pixel = aux.item(i, j, k) - low_pass.item(i, j, k)
                aux.itemset((i, j, k), pixel)

    cv2.imwrite('../output/3.png', aux)

    return aux

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
input = cv2.imread('../input/p1-1-0.png')

# 2.1

filter = []
filter.append(mask.g_3)
filter.append(mask.g_7)
filter.append(mask.g_15)

time = ut.time()

for i in range(len(filter)):
    time = ut.time()
    output = cv.convolve(cp.copy(input), filter[i])
    print("Convolution time[", i, "]: ", time.elapsed())
    cv2.imwrite('../output/p1-2-1-{}.png'.format(i), output)

    time = ut.time()
    output = cv2.filter2D(cp.copy(input), -1, np.flip(np.flip(filter[i], 0), 1))
    print("OpenCV Convolution time[", i, "]: ", time.elapsed())
    print("")

# 2.2

output = gPyr.gaussianPyramid(cp.copy(input), 3)

for i in range(len(output)):
    cv2.imwrite('../output/p1-2-2-{}.png'.format(i), output[i])

# 2.3

output = pPyr.placePyramid(cp.copy(input), 3)

for i in range(len(output)):
    cv2.imwrite('../output/p1-2-3-{}.png'.format(i), output[i])

# 2.3 Reconstruct

aux = output[len(output) - 1]
for j in range(len(output) - 1,  0, -1):
    aux = pPyr.pyrExpand(aux, output[j - 1])

i += 1
cv2.imwrite('../output/report/p1-2-3-reconstruction.png', aux)

# 2.4

img1 = cv2.imread('../input/img1.png')
img2 = cv2.imread('../input/img2.png')
b_mask = cv2.imread('../input/b_mask.png')

output = bl.blend(img1, img2, b_mask, 4)
cv2.imwrite('../output/p1-2-4.png', output)

# 3.2

img1 = cv2.imread('../input/img1.png')
img2 = cv2.imread('../input/img2.png')
b_mask = cv2.imread('../input/b_mask.png')

i1_mask = img1 * (b_mask / 255)
cv2.imwrite('../output/report/freq_img1_mask.png', i1_mask)

i2_mask = img2 * (1 - (b_mask / 255))
cv2.imwrite('../output/report/freq_img2_mask.png', i2_mask)

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

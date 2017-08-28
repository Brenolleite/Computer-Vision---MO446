import cv2
import copy as cp
import numpy as np
import utils as ut

import masks as mask
import convolution as cv
import gaussian_pyramid as gPyr
import place_pyramid as pPyr
import blending as bl
import fourier as ft

# Python uses the BGR color scheme
input = cv2.imread('../input/p1-1-0.png')

# 2.1

filter = [mask.g_3, mask.g_7, mask.g_15]

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

# Image numbering control
k = 0

output = pPyr.placePyramid(cp.copy(input), 3)

for i in range(len(output)):
    cv2.imwrite('../output/p1-2-3-{}.png'.format(k), output[i])
    k += 1

# 2.3 Reconstruct

aux = output[len(output) - 1]
for j in range(len(output) - 1,  0, -1):
    aux = pPyr.pyrExpand(aux, output[j - 1])

i += 1
cv2.imwrite('../output/report/p1-2-3-{}.png'.format(k), aux)
k += 1

# 2.4

img1 = cv2.imread('../input/p1-2-4-0.png')
img2 = cv2.imread('../input/p1-2-4-1.png')
b_mask = cv2.imread('../input/p1-2-4-2.png')

output = bl.blend(img1, img2, b_mask, 4)
cv2.imwrite('../output/p1-2-4.png', output)

# 3.2

img1 = cv2.imread('../input/p1-2-4-0.png')
img2 = cv2.imread('../input/p1-2-4-1.png')
b_mask = cv2.imread('../input/p1-2-4-2.png')

i1_mask = img1 * (b_mask / 255)
cv2.imwrite('../output/report/freq_img1_mask.png', i1_mask)

i2_mask = img2 * (1 - (b_mask / 255))
cv2.imwrite('../output/report/freq_img2_mask.png', i2_mask)

#3.1

# The type of file (0) is necessary in this case
input = cv2.imread('../input/frequency.png', 0)

magnitude, phase = ft.transform(input, True)

cv2.imwrite('../output/phase.png', phase)
cv2.imwrite('../output/magnitude.png', magnitude)

phase_back = ft.reconstruct(magnitude, phase, "phase", 1, "desc", True)

cv2.imwrite('../output/phase_back.png', phase_back)
#cv2.imwrite('../output/magnitude_back.png', magnitude_back)

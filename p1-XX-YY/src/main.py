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

#3.1

# The type of file (0) is necessary in this case
input = cv2.imread('../input/frequency.png', 0)

magnitude, phase = transform(input)

cv2.imwrite('../output/phase.png', phase)
cv2.imwrite('../output/magnitude.png', magnitude)

phase_back = reconstruct(magnitude, phase, "phase", 1, "desc")

cv2.imwrite('../output/phase_back.png', phase_back)
#cv2.imwrite('../output/magnitude_back.png', magnitude_back)

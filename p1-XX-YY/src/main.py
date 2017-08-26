import cv2
import copy as cp
import numpy as np
import utils as ut

import masks as mask
import convolution as cv
import gaussian_pyramid as gPyr
import place_pyramid as pPyr

# Python uses the BGR color scheme
input = cv2.imread('../input/messi5.png')

# 2.1
time = ut.time()
output = cv.convolve(cp.copy(input), mask.g_3)
print("Convolution time:" + time.elapsed())

cv2.imwrite('../output/p1-1-0.png', output)

time = ut.time()

output = cv2.filter2D(cp.copy(input), -1, np.flip(np.flip(mask.g_3, 0), 1))

print("OpenCV Convolution time:" + time.elapsed())

cv2.imwrite('../output/p1-1-1.png', output)

# 2.2
# gPyramidDown = gPyr.gauPyrDown(cp.copy(input), 5)
# gPyramidUp = gPyr.gauPyrUp(cp.copy(input), 3)

# 2.3
output = pPyr.pyrDown(cp.copy(input), 3)
pPyr.pyrUp(output, 3)

import cv2
import copy as cp
import numpy as np
import utils as ut

import masks as mask
import convolution as cv

# Python uses the BGR color scheme
input = cv2.imread('../input/p1-1-2.jpg')

time = ut.time()
output = cv.convolve(cp.copy(input), mask.g_3)
print("Convolution time:" + time.elapsed())

cv2.imwrite('../output/p1-1-0.png', output)


time = ut.time()
cv2.filter2D(cp.copy(input), -1, np.flip(np.flip(mask.g_3, 0), 1), output)
print("OpenCV Convolution time:" + time.elapsed())

cv2.imwrite('../output/p1-1-1.png', output)

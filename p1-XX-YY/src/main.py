import cv2
import copy as cp
import numpy as np

import masks as mask
import convolution as cv

# Python uses the BGR color scheme
input = cv2.imread('../input/input.jpg')

output = cv.convolve(cp.copy(input), mask.g_3)

cv2.imwrite('../output/TESTE.jpg', output)

cv2.filter2D(cp.copy(input), -1, np.flip(np.flip(mask.g_3, 0), 1), output)

cv2.imwrite('../output/TESTE2.jpg', output)

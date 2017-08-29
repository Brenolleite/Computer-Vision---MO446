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
input = cv2.imread('input/p1-1-0.png')

# 2.1

filter = [mask.g_3, mask.g_7, mask.g_15]

time = ut.time()

for i in range(len(filter)):
    time = ut.time()
    output = cv.convolve(cp.copy(input), filter[i])
    print("Convolution time[", i, "]: ", time.elapsed())
    cv2.imwrite('output/p1-2-1-{}.png'.format(i), output)

    time = ut.time()
    output = cv2.filter2D(cp.copy(input), -1, np.flip(np.flip(filter[i], 0), 1))
    print("OpenCV Convolution time[", i, "]: ", time.elapsed())
    print("")

# 2.2

output = gPyr.gaussianPyramid(cp.copy(input), 3)

for i in range(len(output)):
    cv2.imwrite('output/p1-2-2-{}.png'.format(i), output[i])

# 2.3

# Image numbering control
k = 0

output = pPyr.placePyramid(cp.copy(input), 3)

for i in range(len(output)):
    cv2.imwrite('output/p1-2-3-{}.png'.format(k), output[i])
    k += 1

# 2.3 Reconstruct

aux = output[len(output) - 1]
for j in range(len(output) - 1,  0, -1):
    aux = pPyr.pyrExpand(aux, output[j - 1])

i += 1
cv2.imwrite('output/report/p1-2-3-{}.png'.format(k), aux)
k += 1

# 2.4

img1 = cv2.imread('input/p1-2-4-0.png')
img2 = cv2.imread('input/p1-2-4-1.png')
b_mask = cv2.imread('input/p1-2-4-2.png')

output = bl.blend(img1, img2, b_mask, 4)
cv2.imwrite('output/p1-2-4-0.png', output)

img1 = cv2.imread('input/p1-2-4-3.png')
img2 = cv2.imread('input/p1-2-4-4.png')
b_mask = cv2.imread('input/p1-2-4-5.png')

output = bl.blend(img1, img2, b_mask, 4)
cv2.imwrite('output/p1-2-4-1.png', output)

#3.1

# Importing in grayscale for cleaner code
input = cv2.imread('input/p1-1-0.png', 0)

percents = [-1, 25, 50, 75, 100]
index = 2

# Creating changes using incremental and changing phase
for perc in percents:
    magnitude, phase = ft.transform(cp.copy(input), (perc == 100))

    if perc == 100:
        cv2.imwrite('output/p1-3-1-0.png', phase)
        cv2.imwrite('output/p1-3-1-1.png', magnitude)

    img_frequency = ft.reconstruct(magnitude, phase, "phase", perc, "inc", (perc == 100))
    cv2.imwrite('output/p1-3-1-{}.png'.format(index), img_frequency)
    index += 1

# Creating changes using decreasing and changing phase
for perc in percents:
    magnitude, phase = ft.transform(cp.copy(input), (perc == 100))

    img_frequency = ft.reconstruct(magnitude, phase, "phase", perc, "desc", (perc == 100))
    cv2.imwrite('output/p1-3-1-{}.png'.format(index), img_frequency)
    index += 1

# Creating changes using incremental and magnitude phase
for perc in percents:
    magnitude, phase = ft.transform(cp.copy(input), (perc == 100))

    img_frequency = ft.reconstruct(magnitude, phase, "magnitude", perc, "inc", (perc == 100))
    cv2.imwrite('output/p1-3-1-{}.png'.format(index), img_frequency)
    index += 1

# Creating changes using decreasing and changing phase
for perc in percents:
    magnitude, phase = ft.transform(cp.copy(input), (perc == 100))

    img_frequency = ft.reconstruct(magnitude, phase, "magnitude", perc, "desc", (perc == 100))
    cv2.imwrite('output/p1-3-1-{}.png'.format(index), img_frequency)
    index += 1

# Reimport colored image
input = cv2.imread('input/p1-1-0.png')

# Creating changes using decreasing and changing magnitude
# (with color showing that it still working)
for perc in percents:
    img_frequency = np.zeros(input.shape)

    # Execute transformation for each channel
    for i in range(3):
        magnitude, phase = ft.transform(cp.copy(input[:, :, i]), (perc == 100))

        img_frequency[:, :, i] = ft.reconstruct(magnitude, phase, "magnitude", perc, "desc", (perc == 100))
    cv2.imwrite('output/p1-3-1-{}.png'.format(index), img_frequency)
    index += 1

#3.2

# Importing in grayscale for easier implementation
img1 = cv2.imread('input/p1-2-4-0.png')
img2 = cv2.imread('input/p1-2-4-1.png')
b_mask = cv2.imread('input/p1-2-4-2.png')

# Create image
img_blended = np.zeros(img1.shape)

# Executes bleding in each channel
for i in range(3):
    img_blended[:, :, i] = ft.blend_frequencies(img1[:, :, i], img2[:, :, i], b_mask[:, :, i])
cv2.imwrite('output/p1-3-2-0.png', img_blended)

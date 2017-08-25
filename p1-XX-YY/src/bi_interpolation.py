import cv2
import numpy as np
import math
import copy as cp

def interpolate(input):
    output = cp.copy(input)

    if(input.ndim == 2):
        interpolation(input, output, -1)
    else:
        for i in range(3):
            interpolation(input[:, :, i], output, i)

    return output

def interpolation(input, output, channel):
    i_height, i_width = input.shape

    for i in range(1, i_height, 2):
        for j in range(1, i_width, 2):

            # Up
            if (j + 1 < i_width):
                mean = math.floor((input.item(i - 1, j - 1) + input.item(i - 1, j + 1)) / 2)
                output.itemset((i - 1, j, channel), mean)
            else:
                output.itemset((i - 1, j, channel), output.item((i - 1, j - 1, channel)))

            # Left
            if (i + 1 < i_height):
                mean = math.floor((input.item(i - 1, j - 1) + input.item(i + 1, j - 1)) / 2)
                output.itemset((i, j - 1, channel), mean)
            else:
                output.itemset((i, j - 1, channel), output.item((i - 1, j - 1, channel)))

            # Center
            if (i + 1 < i_height) and (j + 1 < i_width):
                mean = math.floor((input.item(i - 1, j - 1) + input.item(i - 1, j + 1) + input.item(i + 1, j - 1) + input.item(i + 1, j + 1)) / 4)
                output.itemset((i, j, channel), mean)
            else:
                output.itemset((i, j, channel), output.item(i - 1, j - 1, channel))

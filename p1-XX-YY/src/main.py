import cv2
import math
import copy as cp
import numpy as np

# Python uses the BGR color scheme

input = cv2.imread('../input/input.jpg')

# Gaussian pyramid

def pyrDown(img, lvls):
    pyramid = np.array

    for i in range(0, lvls):
        i_height, i_width, i_channels = img.shape
        height = math.floor(i_height / 2)
        width = math.floor(i_width / 2)
        newImg = np.zeros((height, width, 3), np.uint8)

        for j in range(0, height):
            for l in range(0, width):
                newImg.itemset((j, l, 0), img.item(j * 2, l * 2, 0))
                newImg.itemset((j, l, 1), img.item(j * 2, l * 2, 1))
                newImg.itemset((j, l, 2), img.item(j * 2, l * 2, 2))

        cv2.imwrite('../output/p1-2-2-{}.jpg'.format(i), newImg)
        img = newImg

# def Access method to obtain a given level
# def Pyramid construction needs the image source and the number of levels, one to go UP and one to go DOWN, maybe I should insert this information in the pyrX methods

pyrDown(cp.copy(input), 3)

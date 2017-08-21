import cv2
import math
import copy as cp
import numpy as np

# Python uses the BGR color scheme

input = cv2.imread('../input/input.jpg')

# Gaussian Mask 3X3
g_mask_3 = np.matrix([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
    ])

# Gaussian Mask 7X7
g_mask_7 = np.matrix([
    [1, 1, 2, 2, 2, 1, 1],
    [1, 3, 4, 5, 4, 3, 1],
    [2, 4, 7, 8, 7, 4, 2],
    [2, 5, 8, 10, 8, 5, 2],
    [2, 4, 7, 8, 7, 4, 2],
    [1, 3, 4, 5, 4, 3, 1],
    [1, 1, 2, 2, 2, 1, 1]
    ])

# Gaussian Mask 15X15
g_mask_15 = np.matrix([
    [2, 2, 3, 4, 5, 5, 6, 6, 6, 5, 5, 4, 3, 2, 2],
    [2, 3, 4, 5, 7, 7, 8, 8, 8, 7, 7, 5, 4, 3, 2],
    [3, 4, 6, 7, 9, 10, 10, 11, 10, 10, 9, 7, 6, 4, 3],
    [4, 5, 7, 9, 10, 12, 13, 13, 13, 12, 10, 9, 7, 5, 4],
    [5, 7, 9, 11, 13, 14, 15, 16, 15, 14, 13, 11, 9, 7, 5],
    [5, 7, 10, 12, 14, 16, 17, 18, 17, 16, 14, 12, 10, 7, 5],
    [6, 8, 10, 13, 15, 17, 19, 19, 19, 17, 15, 13, 10, 8, 6],
    [6, 8, 11, 13, 16, 18, 19, 20, 19, 18, 16, 13, 11, 8, 6],
    [6, 8, 10, 13, 15, 17, 19, 19, 19, 17, 15, 13, 10, 8, 6],
    [5, 7, 10, 12, 14, 16, 17, 18, 17, 16, 14, 12, 10, 7, 5],
    [5, 7, 9, 11, 13, 14, 15, 16, 15, 14, 13, 11, 9, 7, 5],
    [4, 5, 7, 9, 10, 12, 13, 13, 13, 12, 10, 9, 7, 5, 4],
    [3, 4, 6, 7, 9, 10, 10, 11, 10, 10, 9, 7, 6, 4, 3],
    [2, 3, 4, 5, 7, 7, 8, 8, 8, 7, 7, 5, 4, 3, 2],
    [2, 2, 3, 4, 5, 5, 6, 6, 6, 5, 5, 4, 3, 2, 2]
    ])

# Gaussian pyramid

def pyrDown(img, lvls):
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

def pyrUp(img, lvls):
    print("Pyramid UP not implemented yet")

def pyrAccLvl(pyrImg, lvl):
    print("Access to individual lvls not implemented yet")

# Convolute a linear filter to an image
def filter2d(img, kernel, anchor):
    # Image dimensions
    i_height, i_width, i_channels = img.shape

    # Kernel dimensions
    k_height, k_width = kernel.shape

    # Anchor distance to border
    anchor_distance = math.floor(k_height / 2)

    # Apply the filter in every pixel
    for i in range(0, i_height):
        for j in range(0, i_width):
            
            # Operate with kernel
            for g in range(0, k_height):
                for h in range (0, k_width):

def gaussianSomething():
    # Usar a função filter2D para gerar o kernel, http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/filter_2d/filter_2d.html
    gaussian = cv2.getGaussianKernel(9, 1.7)
    cv2.imwrite('../output/gaussian.jpg', gaussian)

# def Access method to obtain a given level

# Execution section

pyrDown(cp.copy(input), 3)
pyrUp(cp.copy(input), 3)
filter2d(cp.copy(input), g_mask_3, 1)
# filter2d(cp.copy(input), g_mask_7, 3)
# filter2d(cp.copy(input), g_mask_15, 7)

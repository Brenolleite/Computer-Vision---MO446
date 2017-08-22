import cv2
import math
import copy as cp
import numpy as np
import os
import masks

# Python uses the BGR color scheme
input = cv2.imread('../input/input.jpg')

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
    print(i_height, " Image Size")

    # Kernel dimensions
    k_height, k_width = kernel.shape

    # Anchor distance to border
    anchor_distance = math.floor(k_height / 2)

    newImg = np.zeros((i_height, i_width, 3), np.uint8)

    # Apply the filter in every pixel
    for i in range(0, i_height):
        for j in range(0, i_width):
            
            # Kernel factor
            k_weight = 0
            sum = 0
            channel_0 = 0
            channel_1 = 0
            channel_2 = 0

            # print("Velho ", i, "-", j, ": ", newImg.item((i, j, 0)))

            # Operate with kernel
            for g in range(0, k_height):
                for h in range (0, k_width):
                    k_weight = 0

                    if i - abs(g - anchor_distance) >= 0 and j - abs(h - anchor_distance) >= 0:
                        print("[", i,"-", j,"] With [", g,"-", h,"]")
                        k_weight = kernel.item((g, h))
                    if i + (g - anchor_distance) < i_height and j + (h - anchor_distance) < i_width:
                        k_weight = kernel.item((g, h))

                    channel_0 += k_weight * img.item((i, j, 0))
                    channel_1 += k_weight * img.item((i, j, 1))
                    channel_2 += k_weight * img.item((i, j, 2))
                    sum += k_weight

            newImg.itemset((i, j, 0), channel_0 / sum)
            newImg.itemset((i, j, 1), channel_1 / sum)
            newImg.itemset((i, j, 2), channel_2 / sum)

            # print("Novo ", i, "-", j, ": ", newImg.item((i, j, 0)))

    cv2.imwrite('../output/TESTE.jpg', newImg)
    print(kernel)
    # print("Begin\n", newImg, "\nEnd")

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


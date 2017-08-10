#!/usr/bin/python3.6

import cv2
import copy as cp
import numpy as np

input = cv2.imread('./input/p0-1-0.png')

# I know that the function cvtColor exists, however as we are new to OpenCV we prefered to learn the hand coded way.

def swapRedBlue(img):
    aux = img[:,:,0]
    img[:,:,0] = img[:,:,2]
    img[:,:,2] = aux

    cv2.imwrite('./output/p0-2-a-0.png', img)
    return img

def monochromeGreen(img):
    img = img[:,:,1]

    cv2.imwrite('./output/p0-2-b-0.png', img)
    return img

def monochromeRed(img):
    img = img[:,:,2]

    cv2.imwrite('./output/p0-2-c-0.png', img)
    return img

def insertImage(imgA, imgB):
    heightA, widthA = imgA.shape
    yiA = int(heightA/2) - 50
    yfA = int(heightA/2) + 50
    xiA = int(widthA/2) - 50
    xfA = int(widthA/2) + 50

    heightB, widthB = imgB.shape
    yiB = int(heightB/2) - 50
    yfB = int(heightB/2) + 50
    xiB = int(widthB/2) - 50
    xfB = int(widthB/2) + 50

    imgB[yiB:yfB, xiB:xfB] = imgA[yiA:yfA, xiA:xfA]

    cv2.imwrite('./output/p0-3-0.png', imgB)
    return imgB

def replaceChannelGreen(imgC, imgD):
    imgD[:,:,1] = imgC[:,:]

    cv2.imwrite('./output/p0-3-1.png', imgD)
    return input

def maxMinMean(img):
    print("Min: {0} Max: {1} Mean: {2} Standard Deviation: {3}".format(np.min(img), np.max(img), np.mean(img), np.std(img)))

def normalize(img):
    mean = np.mean(img)
    deviation = np.std(img)
    img = (((img - mean)/deviation) * 10) + mean

    cv2.imwrite('./output/p0-4-b-0.png', img)

def shiftLeft(img, shift):
    img = np.roll(img, (-1 * shift))
    height, width = img.shape
    img[:,width-shift:] = 0

    cv2.imwrite('./output/p0-4-c-0.png', img)
    return img

def subtractImages(img, imgSub):
    img[:,:] = img[:,:] - imgSub

    cv2.imwrite('./output/p0-4-c-1.png', img)

def addNoise(img, channel, sigma, index):
    height, width, depth = img.shape

    noise = np.random.normal(0, sigma, (height,width))
    img[:,:,channel] = img[:,:,channel] + noise

    cv2.imwrite('./output/p0-5-{0}-0.png'.format(index), img)

swapRedBlue(cp.copy(input))

B = monochromeGreen(cp.copy(input))

A = monochromeRed(cp.copy(input))

C = insertImage(A, cp.copy(B))

replaceChannelGreen(C, cp.copy(input))

maxMinMean(B)

normalize(cp.copy(B))

imgShifted = shiftLeft(cp.copy(B), 2)

subtractImages(cp.copy(B), imgShifted)

sigma = 50

addNoise(cp.copy(input), 1, sigma, 'a')

addNoise(cp.copy(input), 0, sigma, 'b')

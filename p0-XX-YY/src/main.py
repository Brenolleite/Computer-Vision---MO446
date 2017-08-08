#!/usr/bin/python3.6

import cv2
import copy as cp
import numpy as np

input = cv2.imread('../input/p0-1-0.jpg')

# I know that the function cvtColor exists, however as we are new to OpenCV we prefered to learn the hand coded way.

def swapRedBlue(img):
    aux = img[:,:,0]
    img[:,:,0] = img[:,:,2]
    img[:,:,2] = aux
    cv2.imwrite('../output/p0-2-a-0.png', img)
    return img

def monochromeGreen(img):
    img = img[:,:,1]
    cv2.imwrite('../output/p0-2-b-0.png', img)
    return img

def monochromeRed(img):
    img = img[:,:,0]
    cv2.imwrite('../output/p0-2-c-0.png', img)
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
    #imgB[yiB:yfB, xiB:xfB, 0] = imgA[yiA:yfA, xiA:xfA]
    #imgB[yiB:yfB, xiB:xfB, 1] = imgA[yiA:yfA, xiA:xfA]
    #imgB[yiB:yfB, xiB:xfB, 2] = imgA[yiA:yfA, xiA:xfA]

    cv2.imwrite('../output/p0-3-0.png', imgB)

    return imgB

def replaceChannelGreen(imgC, input):
    input[:,:,1] = imgC[:,:]

    cv2.imwrite('../output/p0-3-1.png', input)
    return input

def maxMinMean(img):
    print("Min: {0} Max: {1} Mean: {2} Standard Deviation: {3}".format(np.min(img), np.max(img), np.mean(img), np.std(img)))

def normalize(img):
    mean = np.mean(img)
    deviation = np.std(img)

    img = (((img - mean)/deviation) * 10) + mean

    cv2.imwrite('../output/p0-4-b-0.png', input)

swapRedBlue(cp.copy(input))

imgA = monochromeGreen(cp.copy(input))

imgB = monochromeRed(cp.copy(input))

imgC = insertImage(imgA, cp.copy(imgB))

replaceChannelGreen(imgC, cp.copy(input))

maxMinMean(imgA)

normalize(cp.copy(imgA))

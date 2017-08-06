#!/usr/bin/python3.6

import cv2

input = cv2.imread('../input/p0-1-0.jpg')

# I know that the function cvtColor exists, however as we are new to OpenCV we prefered to learn the hand coded way.

def swapRedBlue(img):
    aux = img[:,:,0]
    img[:,:,0] = img[:,:,2]
    img[:,:,2] = aux
    cv2.imwrite('../output/p0-2-a-0.jpg', img)
    return img

def monochromeGreen(img):
    img[:,:,0] = img[:,:,2] = 0
    cv2.imwrite('../output/p0-2-b-0.jpg', img)
    return img

def monochromeRed(img):
    img[:,:,1] = img[:,:,2] = 0
    cv2.imwrite('../output/p0-2-c-0.jpg', img)

imgA = swapRedBlue(input)

imgB = monochromeGreen(input)

monochromeRed(input)



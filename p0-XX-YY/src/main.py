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
    img = img[:,:,2]
    cv2.imwrite('../output/p0-2-b-0.jpg', img)
    return img

def monochromeRed(img):
    img = img[:,:,2]
    cv2.imwrite('../output/p0-2-c-0.jpg', img)

def insertImage(imgA, imgB):
    heightA, widthA = imgA.shape
    yiA = int(heightA/2) - 50
    yfA = int(heightA/2) + 50
    xiA = int(widthA/2) - 50
    xfA = int(widthA/2) + 50

    heightB, widthB, depth = imgB.shape
    yiB = int(heightB/2) - 50
    yfB = int(heightB/2) + 50
    xiB = int(widthB/2) - 50
    xfB = int(widthB/2) + 50

    imgB[yiB:yfB, xiB:xfB, 0] = imgA[yiA:yfA, xiA:xfA]
    imgB[yiB:yfB, xiB:xfB, 1] = imgA[yiA:yfA, xiA:xfA]
    imgB[yiB:yfB, xiB:xfB, 2] = imgA[yiA:yfA, xiA:xfA]

    cv2.imwrite('../output/p0-3-0.jpg', imgB)

imgA = swapRedBlue(input)

imgB = monochromeGreen(input)

monochromeRed(input)

insertImage(imgB, input)

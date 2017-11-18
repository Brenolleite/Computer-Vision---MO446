import cv2
import numpy as np
import copy as cp

import background as bg

param2_ = 0
param2_max = 255

def find(frame):
    # Copy frame
    img = cp.copy(frame)

    # Check if image is in grayscale
    if len(img.shape) > 2:
        # Transform to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur image to use in hough circles
    img = cv2.medianBlur(img, 5)

    # Find circles using hough
    circles = None

    param = 25
    while(type(circles) != np.ndarray and param > 10):
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, 1, 10, param2 = param, minRadius = 0, maxRadius = 0)
        #  print(param)
        param -= 1

    if type(circles) == np.ndarray:
       # Round positions to grid image
        circles = np.uint16(np.around(circles))

    return circles

def draw(frame, circles):
    # Copy frame to keep it safe
    img = cp.copy(frame)

    # For each circles
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img, (i[0],i[1]), i[2], (0,255,0), 2)
        # draw the center of the circle
        cv2.circle(img, (i[0],i[1]), 2, (0,0,255), 3)

    return img


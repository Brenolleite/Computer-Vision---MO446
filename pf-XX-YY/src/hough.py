import cv2
import numpy as np
import copy as cp

import background as bg

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

    param = 100

    circles = []
    while(param > 10 and len(circles) < 2):
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, 1, 10, param2 =
                param, minRadius = 0, maxRadius = 0)
        #  print(param)
        param -= 1

        if type(circles) != np.ndarray:
            circles = []
        else:
            circles = circles[0,:]

    if type(circles) == np.ndarray:
        # Round positions to grid image
        circles = np.uint16(np.around(circles))

    return circles

def draw(frame, circles):
    # Copy frame to keep it safe
    img = cp.copy(frame)

    # For each circles
    for i in circles:
        # draw the outer circle
        cv2.circle(img, (i[0],i[1]), i[2], (0,255,0), 2)
        # draw the center of the circle
        cv2.circle(img, (i[0],i[1]), 2, (0,0,255), 3)

    return img

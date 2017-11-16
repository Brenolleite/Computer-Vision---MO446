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
    treshold = 60
    while(type(circles) != list and treshold > 0):
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT, 1, 20,
                                   param1 = 5, param2 = treshold, minRadius = 0, maxRadius = 0)
        print(treshold)
        treshold -= 20

    if type(circles) == list:
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

# Getting video and information
video = cv2.VideoCapture('../input/blue.mp4')
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

back = bg.Background()

for i in range(1):
    ret, colorFrame = video.read()

    #sub_img = back.subtraction(colorFrame)

    circles = find(colorFrame)

    if type(circles) == list:
        colorFrame = draw(colorFrame, circles)

    cv2.imwrite('../output/' + str(i) + '.png', colorFrame)
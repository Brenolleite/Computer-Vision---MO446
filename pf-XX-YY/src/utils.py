import time as t
import math
import numpy as np
import copy as cp
import cv2
from random import randint

class Time:
    # Measuring time
    def __init__(self):
        self.start_time = t.time()

    def elapsed(self):
        return t.time() - self.start_time


# Save kmeans colored image
def k_image(img, center, label):
    center = np.uint8(center)
    aux = center[label.flatten()]
    kimg = aux.reshape((img.shape))

    return kimg

# Create image from components
def components_image(comps):
    colors = []

    # Define some colors manually
    colors.append((0,255,0))
    colors.append((0,0,255))
    colors.append((255,0,0))
    colors.append((255,0,255))

    # Create random colors
    for i in np.unique(comps) - len(colors):
        colors.append(((randint(0, 255)), randint(0, 255), randint(0, 255)))

    # Create image
    height, width = comps.shape
    img = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):
            val = comps[y,x]
            if val == -1:
                img[y,x] = (255,255,255)
            else:
                img[y,x] = colors[val]

    return img.astype(np.uint8)

# Draw the bounding boxes
def drawBoundingBox(img, regions):
    # Goes over all found regions
    for i in range(len(regions)):
        # Set the left upper corner of the square
        pt1 = (regions[i][4][0], regions[i][4][2])

        # Same thing for the right downer
        pt2 = (regions[i][4][1], regions[i][4][3])

        # Set a color for the bouding box
        color = ((randint(0, 255), randint(0, 255), randint(0, 255)))

        # Draw the rectangle
        cv2.rectangle(img, pt1, pt2, color, thickness=2, lineType=8, shift=0)

    return img

def drawCentroids(img, regions):
    for k in range(len(regions)):
        center = (int(regions[k][3][0]), int(regions[k][3][1]))

        # Set a color for the bouding box
        color = ((randint(0, 255), randint(0, 255), randint(0, 255)))

        cv2.circle(img, center, 10, color, thickness=-1, lineType=8, shift=0)

    return img

def drawBallTrace(frame, coord, ballTrace):
    x, y = coord

    ballTrace.append(coord)
    if (len(ballTrace) > 100):
        ballTrace.pop(0)

    for i in range(len(ballTrace)):
        cv2.circle(frame, (int(ballTrace[i][0]), int(ballTrace[i][1])), 1, (0, 0, 255), -1)

    return (frame, ballTrace)

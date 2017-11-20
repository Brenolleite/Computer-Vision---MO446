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

# Create dict of colors
colors = {}
def drawBallBox(frame, bBox, diff_colors = False):
    global colors

    for i in range(len(bBox)):
        color, box_x1, box_y1, box_x2, box_y2, cen_x, cen_y, b_id = bBox[i]

        # Create new color for new ID
        if b_id not in colors:
            colors[b_id] = randint(0, 180)

        if diff_colors:
            use_color = colors[b_id]
        else:
            use_color = color

        bgr = cv2.cvtColor(np.uint8([[[use_color, 255, 127]]]), cv2.COLOR_HSV2BGR)[0][0]

        aux = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), aux, thickness = 2)
        cv2.circle(frame, (cen_x, cen_y), 2, aux, thickness = -1)

    return frame

# Euclidian distance
def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

# Create random color
def random_color():
    return (randint(0, 255), randint(0, 255), randint(0, 255))

# Parse the circles info into the keypoints that will be used in motion flow
def parseCentroidInfo(balls):
    output = []
    for i in balls:
        output.append([i[5], i[6]])

    return np.float32(output)

# Draw the motion flow from the keypoints returned by LucasKanade
def drawMotionFlow(frame, ballsTrace, BGR = (0, 255, 0)):
    for i in range(len(ballsTrace)):
        frame = drawPoints(frame, ballsTrace[i], BGR)

    return frame

# Draw points
def drawPoints(frame, p, BGR = (255, 0, 255), size = 2):
    # Transform to ndarray
    p = np.array(p)

    # Print all the points
    for i in range(len(p)):
        x = math.floor(p[i, 0])
        y = math.floor(p[i, 1])

        frame = cv2.circle(frame, (x, y), size, BGR, -1)

    return frame

def maintain_size(vec, size):
    # Start vector
    s = len(vec)-size

    # If zeros is negative
    if  s < 0:
        s = 0

    # Return last elements of vector
    # after starting point
    return vec[s:]

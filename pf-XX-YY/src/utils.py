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

def drawBallBox(frame, bBox):
    for i in range(len(bBox)):
        color, box_x1, box_y1, box_x2, box_y2, cen_x, cen_y, _ = bBox[i]

        bgr = cv2.cvtColor(np.uint8([[[color, 255, 127]]]), cv2.COLOR_HSV2BGR)[0][0]

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

def drawBallTrace(traceBalls, ballsInfo, frame):
    # Create list of already chosen balls
    #removedBalls = []

    # Loop over current frame balls position
    for cur_pos in np.array(ballsInfo)[:,5:7]:
        # Start min distance
        min_dist = 10000000
        min_idx = -1

        # Go over all the balls from previous frames
        for idx, ball in enumerate(traceBalls):
            # Get last position of ball vectors
            last_pos = ball[1][-1]

            # Find distance
            d = dist(cur_pos, last_pos)

            # Update minimun
            if min_dist > d:
                min_dist = d
                min_idx = idx

        # Insert new position to traceBalls vector
        traceBalls[min_idx][1].append(cur_pos)

    # Create new color
    if len(ballsInfo) > len(traceBalls):
        traceBalls.append(random_color(), [])

    #removedBalls.append((min_idx, min_dist))

    # if len(removedBalls) != len(traceBalls):

    # Draw trace for the balls

    # For over all the balls
    for ball in traceBalls:
        # Get ball color
        color = ball[0]

        # Loop over all the positions (same ball)
        for pos in ball[1]:
            x, y = pos
            cv2.circle(frame, (x, y), 2, (0, 0, 255), thickness = -1)
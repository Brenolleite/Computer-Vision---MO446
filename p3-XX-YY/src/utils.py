import time as t
import math
import cv2
import random

class time:
    # Measuring time

    def __init__(self):
        self.start_time = t.time()

    def elapsed(self):
        return t.time() - self.start_time

def drawKeypoints(frame, kp):


    for i in range(0, len(kp), 1):
        x = math.floor(kp[i][0])
        y = math.floor(kp[i][1])

        B = random.randrange(0, 255)
        G = random.randrange(0, 255)
        R = random.randrange(0, 255)

        frame = cv2.circle(frame, (x, y), 2, (B, G, R))

    return frame

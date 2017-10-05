import numpy as np
import cv2
import math

def sift(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kst = sift.detect(img, None)

    kp = []
    for i in range(len(kst)):
        x = math.floor(kst[i].pt[0])
        y = math.floor(kst[i].pt[1])
        kp.append((x, y))

    return np.array(kp)

def harris(img):
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners = 100,
                          qualityLevel = 0.3,
                          minDistance = 7,
                          blockSize = 7)

    # Remove single dimention to add on table
    return cv2.goodFeaturesToTrack(img, mask = None, **feature_params).squeeze()

# DEBUG
#  video = cv2.VideoCapture('../input/input.mp4')
#  fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#  ret, frame = video.read()
#  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#  kp(frame)

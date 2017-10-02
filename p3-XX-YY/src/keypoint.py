import numpy as np
import cv2
import math

# Return a list of tuples [(x, y)] of each keypoint
def kp(gray):
    print("Selecting Keypoints")

    # If changing this, also change line:
    # frame1 = np.float32(frame1)
    # in KLT.py, current line 95, 118 and 126
    return sift(gray)
    #  return harris(gray)

def sift(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kst = sift.detect(img, None)

    kp = []
    for i in range(len(kst)):
        x = math.floor(kst[i].pt[0])
        y = math.floor(kst[i].pt[1])
        kp.append((x, y))

    return kp

def harris(img):
    dst = cv2.cornerHarris(img, 2, 3, 0.04)

    kp = []
    threshold = 0.01 * dst.max()
    threshold = -255
    for i in range(dst.shape[1]):
        for j in range(dst.shape[0]):
            if dst[j][i] > threshold:
                kp.append((j, i))

    return kp


# DEBUG
#  video = cv2.VideoCapture('../input/input.mp4')
#  fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#  ret, frame = video.read()
#  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#  kp(frame)

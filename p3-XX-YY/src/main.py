import cv2
import utils as ut
import copy as cp
import numpy as np
import math

import keypoint as keypoint
import KLT as motion


video_path = 'input/p3-1-0.mp4'

# 3 Keypoint Selection
video = cv2.VideoCapture(video_path)
ret, colorFrame = video.read()

# Transform the frame in grayscale
frame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2GRAY)

# Metrics for the avarage time
harrisSum = 0.0
siftSum = 0.0
average = 1

for i in range(average):
    t = ut.Time()
    kp1 = keypoint.harris(cp.copy(frame))
    harrisSum += t.elapsed()

    t = ut.Time()
    kp2 = keypoint.sift(cp.copy(frame))
    siftSum += t.elapsed()

# Creating image from keypoints using Harris
img1 = ut.drawKeypoints(colorFrame, np.array([kp1]), (255, 0 ,255), 3)
cv2.imwrite('output/p3-3-0.png', img1)

# Creating image from keypoints using SIFT
img2 = ut.drawKeypoints(colorFrame, np.array([kp2]), (255, 0 ,255), 3)
cv2.imwrite('output/p3-3-1.png', img2)

print("Harris selector average time: ", harrisSum/average)
print("SIFT selector average time: ", siftSum/average)

video.release()

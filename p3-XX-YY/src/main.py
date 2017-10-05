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

frame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2GRAY)

harrisSum = 0.0
siftSum = 0.0
average = 5
for i in range(average):
    t = ut.Time()
    kp = keypoint.harris(cp.copy(frame))
    harrisSum += t.elapsed()
    img = ut.drawKeypoints(colorFrame, np.array([kp]), (255, 0 ,255), 3)
    cv2.imwrite('output/p3-3-0.png', img)

    t = ut.Time()
    kp = keypoint.sift(cp.copy(frame))
    siftSum += t.elapsed()
    img = ut.drawKeypoints(colorFrame, np.array([kp]), (255, 0 ,255), 3)
    cv2.imwrite('output/p3-3-1.png', img)

print("Harris selector average time: ", harrisSum/average)
print("SIFT selector average time: ", siftSum/average)

video.release()

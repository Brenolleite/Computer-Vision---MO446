import cv2
import utils as ut
import copy as cp
import numpy as np
import math

import keypoint as kp
import KLT as motion


video_path = 'input/p3-1-0.mp4'

# 3 Keypoint Selection
video = cv2.VideoCapture(video_path)
ret, frame = video.read()

kp = kp.harris(frame)
img = ut.drawKeypoints(frame, kp, (255, 0 ,255), 3)
cv2.imwrite('output/p3-3-0.png', img)

kp = kp.sift(frame)
img = ut.drawKeypoints(frame, kp, (255, 0 ,255), 3)
cv2.imwrite('output/p3-3-1.png', img)

video.release()

import cv2
import utils as ut
import copy as cp
import numpy as np
import math
import keypoint as keypoint
import KLT as motion
import opencv


# 3 Keypoint Selection
video = cv2.VideoCapture('input/p3-1-0.mp4')
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


# 4 Feature Tracking
video_path = '../input/p3-1-0.mp4'

# Optical flow with our implementation
t = ut.Time()
kps = motion.KLT(video_path)
ut.videoFlow(kps, video_path, '../output/p3-4-0.avi', (102, 255, 102))
print("Our Optical Flow Time: ", t.elapsed())

t = ut.Time()
kps_opencv = opencv.KLT(video_path)
ut.videoFlow(kps, video_path, '../output/p3-4-1.avi', (102, 255, 102))
print("Opencv Optical Flow Time: ", t.elapsed())

# Getting some frames from video
video = cv2.VideoCapture(video_path)
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
count = int(length/2)
img_n = 0
for i in range(length):
    ret, frame = video.read()

    if i == 0 or i == length - 1 or count == 0:
        # Drawing image to our implementation
        frame = ut.drawKeypoints(frame, np.array([kps[i]]), (102, 255, 102), 4)

        cv2.imwrite('output/p3-4-2-{}.png'.format(img_n), frame)

        # Drawing image to opencv implementation
        frame = ut.drawKeypoints(frame, np.array([kps_opencv[i]]), (102, 255, 102), 4)

        cv2.imwrite('output/p3-4-3-{}.png'.format(img_n), frame)

        count = int(length/2)
        img_n += 1

    count -= 1

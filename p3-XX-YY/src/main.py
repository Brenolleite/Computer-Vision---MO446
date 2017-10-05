import cv2
import utils as ut
import copy as cp
import numpy as np
import math
import keypoint as keypoint
import KLT as motion
import opencv
import SFM as struc
import meshlab as ml


# 3 Keypoint Selection
video = cv2.VideoCapture('input/p3-1-0.mp4')
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

print("\n")

# 4 Feature Tracking
video_path = 'input/p3-1-0.mp4'

# Optical flow with our implementation
t = ut.Time()
kpsh = motion.KLT(video_path, 'harris')
print("Our Optical Flow Time (Harris): ", t.elapsed())

t = ut.Time()
kpsh_opencv = opencv.KLT(video_path, 'harris')
print("Opencv Optical Flow Time (Harris): ", t.elapsed())

ut.videoFlow2(kpsh, kpsh_opencv,  video_path, 'output/p3-4-0.avi', (102, 255, 102), (144, 16, 242))

print("\n")

t = ut.Time()
kpss = motion.KLT(video_path, 'sift')
print("Our Optical Flow Time (SIFT): ", t.elapsed())

t = ut.Time()
kpss_opencv = opencv.KLT(video_path, 'sift')
print("Opencv Optical Flow Time (SIFT): ", t.elapsed())

ut.videoFlow2(kpss, kpss_opencv,  video_path, 'output/p3-4-1.avi', (102, 255, 102), (144, 16, 242))


# Getting some frames from video
video = cv2.VideoCapture(video_path)
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

img_n = 0
for i in range(length - 1):
    ret, frame = video.read()

    if i == 0 or i == length - 2 or i == int(length/2):
        # Drawing image to our implementation
        frame1 = ut.drawKeypoints(cp.copy(frame), np.array([kpsh[i]]), (102, 255, 102), 4)

        cv2.imwrite('output/p3-4-4-{}.png'.format(img_n), frame1)

        # Drawing image to opencv implementation
        frame2 = ut.drawKeypoints(cp.copy(frame), np.array([kpsh_opencv[i]]), (144, 16, 242), 4)

        cv2.imwrite('output/p3-4-5-{}.png'.format(img_n), frame2)

        img_n += 1

print("\n")

# 5 Structure From Motion
kps = opencv.KLT('input/p3-1-1.mp4', 'sift')
points, colors, cam_points, cam_colors = struc.sfm(kps)

# Filter points (outliers) for better visualization
points = points[np.where(points[:,0] < - 50)]
colors = colors[np.where(points[:,0] < - 50)]

# Writing meshlab file
ml.write_ply('output/p3-5-0.ply', points, colors)
ml.write_ply('output/p3-5-1.ply', cam_points, cam_colors)
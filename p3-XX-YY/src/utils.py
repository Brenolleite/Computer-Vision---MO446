import time as t
import math
import numpy as np
import copy as cp
import cv2
import random

class Time:
    # Measuring time
    def __init__(self):
        self.start_time = t.time()

    def elapsed(self):
        return t.time() - self.start_time

def drawKeypoints(frame, kps, BGR, size):
    for i in range(len(kps)):
        for j in range(len(kps[i])):
            x = math.floor(kps[i, j, 0])
            y = math.floor(kps[i, j, 1])

            frame = cv2.circle(frame, (x, y), size, BGR, -1)

    return frame

#
# kps      -> Matrix with Frames x Keypoints
# path     -> Input video path
# out_path -> Output video path
# BGR      -> (B, G, R) colors
#
def videoFlow(kps, path, out_path, BGR):
    # Transform into numpy array
    kps = np.array(kps)

    # Open video and get settings
    video = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = video.get(cv2.CAP_PROP_FPS)

    # Open output video
    output_video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Draw new images
    for i in range(length):
        # Get frame
        ret, frame = video.read()

        # Draw keypoints
        frame = drawKeypoints(frame, kps[0:i,:,:], BGR, 1)

        # Write new image to video
        output_video.write(frame)

    # Close videos
    video.release()
    output_video.release()

def videoFlow2(kps1, kps2, path, out_path, BGR1, BGR2):
    # Transform into numpy array
    kps1 = np.array(kps1)
    kps2 = np.array(kps2)

    # Open video and get settings
    video = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = video.get(cv2.CAP_PROP_FPS)

    # Open output video
    output_video = cv2.VideoWriter(out_path, fourcc, fps, (2*width, height))

    # Draw new images
    for i in range(length):
        # Get frame
        ret, frame = video.read()

        # Draw keypoints 1
        frame1 = drawKeypoints(cp.copy(frame), kps1[0:i,:,:], BGR1, 1)

        # Draw keypoints 2
        frame2 = drawKeypoints(cp.copy(frame), kps2[0:i,:,:], BGR2, 1)

        out = np.concatenate((frame1, frame2), axis=1)

        # Write new image to video
        output_video.write(out)

    # Close videos
    video.release()
    output_video.release()
import time as t
import math
import numpy as np
import cv2
import random

class Time:
    # Measuring time
    def __init__(self):
        self.start_time = t.time()

    def elapsed(self):
        return t.time() - self.start_time

def drawKeypoints(frame, kps, BGR):
    for i in range(len(kps)):
        for j in range(len(kps[i])):
            x = math.floor(kps[i, j, 0])
            y = math.floor(kps[i, j, 1])

            frame = cv2.circle(frame, (x, y), 1, BGR, -1)

    return frame

def videoFlow(kps, path, out_path, BGR):
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
        frame = drawKeypoints(frame, kps[0:i,:,:], BGR)

        # Write new image to video
        output_video.write(frame)

    # Close videos
    video.release()
    output_video.release()
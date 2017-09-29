import numpy as np

import cv2

# KeyPoins, Last Frame, new Frame
def solver(kp, frame1, frame2, neigh):
    # Concatenate frames
    frames = []
    frames.append(frame1)
    frames.append(frame2)

    # Find derivatives of frames
    It = np.diff(frames, 1, axis=0)
    Iy = np.diff(frames, 1, axis=1)
    Ix = np.diff(frames, 1, axis=2)

    print(It[0].shape , frame1.shape)

# Creating video
video = cv2.VideoCapture('../input/input.mp4')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

ret, frame1 = video.read()
ret, frame2 = video.read()

solver([], frame1, frame2, 15)
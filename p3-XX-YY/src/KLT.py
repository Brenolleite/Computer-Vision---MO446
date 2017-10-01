import keypoint

import copy as cp
import numpy as np
from numpy.linalg import inv

import cv2
import math

# KeyPoins, Last Frame, new Frame
def solver(kp, frame1, frame2, nb):
    print("KeyPoint Flows")

    # Create list of flows
    flows = []

    # Concatenate frames
    frames = []
    frames.append(frame1)
    frames.append(frame2)

    # Find derivatives of frames
    It = np.diff(frames, 1, axis=0)
    Iy = np.diff(frames, 1, axis=1)
    Ix = np.diff(frames, 1, axis=2)

    # Solve u, v for each keypoint
    for i in range(len(kp)):
        # Create matrixes and get kp position
        x = kp[i][0]
        y = kp[i][1]
        nbOffset = math.floor(nb / 2)
        A = []
        d = []
        b = []

        # Getting neighbourhood
        for k in range(x - nbOffset, x + nbOffset + 1, 1):
            for m in range(y - nbOffset, y + nbOffset + 1, 1):
                # Creating matrix A
                A.append([Ix[0,k,m], Iy[0,k,m]])

                # Creating matrix b
                b.append([It[0,k,m]])

        # Execute solver's algebra
        A = np.array(A)
        b = np.array(b)
        At = np.transpose(A)

        i = inv(np.dot(At, A))

        d = -1 * np.dot(At,b)
        d = np.dot(i, d)

        # Adding u,v to kp
        flows.append((d[0,0], d[1,0]))

    # Returning (u,v) vector
    return np.array(flows)

# Eliminate keypoints too close to the border
def filterBorderKeypoints(kp, borderSize, imgSize):
    print("Filtering keypoints")

    for i in range(len(kp) - 1, -1, -1):
        x, y = kp[i]
        if kp[i][0] > imgSize[0] - borderSize or kp[i][0] < borderSize:
            kp.pop(i)
        elif kp[i][1] > imgSize[1] - borderSize or kp[i][1] < borderSize:
            kp.pop(i)

    return kp

def KLT(video):
    print("Executing KLT")

    # Video length, frame count
    length = int(video.get(7))

    # Size of the outlier border for keypoints
    filterBorder = 30

    # Size of the neighbourhood to be accounted in KLT
    solverNeighbourhood = 15

    # For every video frame
    for i in range(0, length, 2):
        print("\nFrames: ", i, i+1)

        # Get frames
        ret, frame1 = video.read()
        ret, frame2 = video.read()

        # Transform frame to grayscale
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame1 = np.float32(frame1)

        # Transform frame to grayscale
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2 = np.float32(frame2)

        # Get keypoints
        kp = keypoint.kp(cp.copy(frame1))

        # Filter keypoints
        kp = filterBorderKeypoints(kp, filterBorder, frame1.shape)

        # Find optical flow
        flows = solver(kp, frame1, frame2, solverNeighbourhood)

        return kp, flows

# DEBUG
video = cv2.VideoCapture('../input/input.mp4')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

KLT(video)

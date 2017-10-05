import keypoint
import utils
from numpy.linalg import lstsq
import copy as cp
import numpy as np
from numpy.linalg import inv

import cv2
import math

# KeyPoins, Last Frame, new Frame
def solver(kp, frame1, frame2, nb):
    # Create list of flows
    flows = []

    # Concatenate frames
    frames = []
    frames.append(frame1)
    frames.append(frame2)

    # Find derivatives of frames
    It = frame2 - frame1
    Iy = np.diff(frame1, 1, axis=0)
    Ix = np.diff(frame1, 1, axis=1)

    # Solve u, v for each keypoint
    size = len(kp)
    for i in range(size - 1, -1, -1):
        # Create matrixes and get kp position
        x = int(kp[i][0])
        y = int(kp[i][1])
        nbOffset = math.floor(nb / 2)
        A = []
        d = []
        b = []

        # Getting neighbourhood
        for k in range(y - nbOffset, y + nbOffset + 1, 1):
            for m in range(x - nbOffset, x + nbOffset + 1, 1):
                # Creating matrix A
                A.append([Ix[k,m], Iy[k,m]])

                # Creating matrix b
                b.append([It[k,m]])

        # Execute least square
        d = np.array(lstsq(A, -np.array(b)))
        flows.append((d[0][0,0], d[0][1,0]))

    # Returning (u,v) vector
    return (np.array(kp), np.array(flows))

# Eliminate keypoints too close to the border
def filterBorderKeypoints(kp, kps, borderSize, imgSize):
    indexes = []
    for i in range(len(kp) - 1, -1, -1):
        x, y = kp[i]

        heigth = imgSize[0]
        width =  imgSize[1]

        if kp[i][0] > width - borderSize or kp[i][0] < borderSize or kp[i][1] > heigth - borderSize or kp[i][1] < borderSize:
            kp = np.delete(kp, i, 0)
            indexes.append(i)

    kps = np.delete(kps, indexes, axis=1)

    return kp, kps


# Transform the keypoints into the new coordinates using the (u, v) flows
# and interpolate the keypoints into valid coordinates
def update_kp(kp, flows):
    new_kp = []
    # Run over all the keypoints
    for i in range(len(kp)):
        x, y = kp[i]
        u, v = flows[i]

        x = math.floor(x + u)
        y = math.floor(y + v)

        new_kp.append((x, y))

    return np.array(new_kp)

def KLT(video_path):
    # Open video and get number of frames
    video = cv2.VideoCapture(video_path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Stores the flow for the ith frame in the ith index
    output = []

    # Size of the outlier border for keypoints
    filterBorder = 30

    # Size of the neighbourhood to be accounted in KLT
    solverNeighbourhood = 15

    # Get frames
    ret, colorFrame1 = video.read()

    # Transform frame to grayscale
    frame1 = np.float32(cv2.cvtColor(colorFrame1, cv2.COLOR_BGR2GRAY))

    # Get keypoints
    kp = keypoint.harris(cp.copy(frame1))

    # Add KP to frames matrix
    output.append(kp)

    # Filter keypoints
    kp, output = filterBorderKeypoints(kp, output, filterBorder, frame1.shape)

    # For every video frame
    for i in range(length - 1):
        # Get frames
        ret, colorFrame2 = video.read()

        # Transform frame to grayscale
        frame2 = np.float32(cv2.cvtColor(colorFrame2, cv2.COLOR_BGR2GRAY))

        # Find optical flow
        kp, flows = solver(kp, frame1, frame2, solverNeighbourhood)

        # Update keypoints
        kp = update_kp(kp, flows)

        # Filter keypoints after update
        kp, output = filterBorderKeypoints(kp, output, filterBorder, frame1.shape)

        # Verify if exists keypoint
        if len(kp) > 0:
            # Add updated KP to frames matrix
            output = np.append(output, [kp], axis=0)

        frame1 = frame2
        colorFrame1 = colorFrame2

    return np.array(output)

video_path = '../input/teste.mp4'
kps = KLT(video_path)
utils.videoFlow(kps, video_path, '../output/flow.avi', (13, 94, 1))

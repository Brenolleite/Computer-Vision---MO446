import keypoint
import utils as ut

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
    size = len(kp)
    for i in range(size - 1, -1, -1):
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

        inverted = None
        try:
            inverted = inv(np.dot(At, A))
        except np.linalg.linalg.LinAlgError as err:
            kp = np.delete(kp, i, 0)
            continue

        d = -1 * np.dot(At,b)
        d = np.dot(inverted, d)

        # Adding u,v to kp
        flows.append((d[0,0], d[1,0]))

    # Returning (u,v) vector
    return (np.array(kp), np.array(flows))

# Eliminate keypoints too close to the border
def filterBorderKeypoints(kp, borderSize, imgSize):
    print("Filtering keypoints")

    size = len(kp)
    for i in range(size - 1, -1, -1):
        x, y = kp[i]
        if kp[i][0] > imgSize[0] - borderSize or kp[i][0] < borderSize:
            kp = np.delete(kp, i, 0)
        elif kp[i][1] > imgSize[1] - borderSize or kp[i][1] < borderSize:
            kp = np.delete(kp, i, 0)

    return np.array(kp)

# Transform the keypoints into the new coordinates using the (u, v) solutions
# and interpolate the keypoints into valid coordinates
def interpolate(kp, solution):

    interpolated = []
    # Run over all the keypoints
    for i in range(len(kp)):
        x = kp[i][0]
        y = kp[i][1]

        u = solution[i][0]
        v = solution[i][1]

        x = math.floor(x + u)
        y = math.floor(y + v)

        interpolated.append((x, y))

    return np.array(interpolated)

def KLT(video, fourcc):
    print("Executing KLT")

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = video.get(cv2.CAP_PROP_FPS)
    
    # Stores the flow for the ith frame in the ith index
    output = []

    # Size of the outlier border for keypoints
    filterBorder = 30

    # Size of the neighbourhood to be accounted in KLT
    solverNeighbourhood = 15

    # Get frames
    ret, colorFrame1 = video.read()

    # Transform frame to grayscale
    frame1 = cv2.cvtColor(colorFrame1, cv2.COLOR_BGR2GRAY)
    
    # HARRIS NEEDS THIS, SIFT WON'T WORK WITH THIS
    #  frame1 = np.float32(frame1)

    # Get keypoints
    kp = keypoint.kp(cp.copy(frame1))

    # Filter keypoints
    kp = filterBorderKeypoints(kp, filterBorder, frame1.shape)

    output.append(kp)

    # For every video frame
    for i in range(0, length - 1, 1):
        # Get frames
        ret, colorFrame2 = video.read()


        # SIFT won't work with this
        #  frame1 = np.float32(frame1)

        # Transform frame to grayscale
        frame2 = cv2.cvtColor(colorFrame2, cv2.COLOR_BGR2GRAY)

        # SIFT won't work with this
        #  frame2 = np.float32(frame2)

        # Find optical flow
        kp, flows = solver(kp, frame1, frame2, solverNeighbourhood)

        # Interpolate
        kp = interpolate(kp, flows)

        # Filter keypoints
        kp = filterBorderKeypoints(kp, filterBorder, frame1.shape)

        output.append(kp)

        frame1 = frame2
        colorFrame1 = colorFrame2

    return np.array(output)

# DEBUG
#  video = cv2.VideoCapture('../input/input5.mp4')
#  fourcc = cv2.VideoWriter_fourcc(*'DIVX')

#  KLT(video, fourcc)

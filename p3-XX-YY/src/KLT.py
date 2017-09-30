import keypoint

import copy as cp
import numpy as np
import cv2
import math

# KeyPoins, Last Frame, new Frame
# Solves the AD = B equation
def solver(kp, frame1, frame2, nb):

    # Solve u, v for each keypoint
    for i in range(len(kp)):
        print(kp[i][0], kp[i][1])
        x = kp[i][0]
        y = kp[i][1]

        nbOffset = math.floor(nb / 2)
        
        print("Center: ", x, y)
        # Montando frame1 e frame2 para frames.append(1) frames.append(2)        
        # For each pixel in the neighbourhood

        #  neigh1 = np.array()
        #  neigh2 = np.array()

        row1 = []
        row2 = []

        # For each pixel in the neighbourhood
        for k in range(x - nbOffset, x + nbOffset, 1):
            subRow1 = []
            subRow2 = []

            for m in range(y - nbOffset, y + nbOffset, 1):
                #  print("Area: ", k, m)

                subRow1.append(frame1[k][m])
                subRow2.append(frame2[k][m])

            row1.append(subRow1)
            row2.append(subRow2)
        neigh1 = np.array(row1)
        neigh2 = np.array(row2)

        frames = []
        frames.append(neigh1)
        frames.append(neigh2)
        
        It = np.diff(frames, 1, axis=0)
        Iy = np.diff(frames, 1, axis=1)
        Ix = np.diff(frames, 1, axis=2)

        print(type(Ix))
        print(Ix.shape)
        print(Ix)
    
    # Concatenate frames
    #  frames = []
    #  frames.append(frame1)
    #  frames.append(frame2)

    #  # Find derivatives of frames
    #  It = np.diff(frames, 1, axis=0)
    #  Iy = np.diff(frames, 1, axis=1)
    #  Ix = np.diff(frames, 1, axis=2)

    #  print(It[0].shape , frame1.shape)

    return []

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

# START HERE
# Wrapper function
def KLT(video):
    print("Executing KLT")

    # Video length, frame count
    length = int(video.get(7))

    # DEBUG
    length = 2

    # Size of the outlier border for keypoints
    filterBorder = 30

    # Size of the neighbourhood to be accounted in KLT
    solverNeighbourhood = 15

    # For every video frame
    for i in range(0, length, 2):
        print("\nProgress: ", i, length)

        ret, frame1 = video.read()
        ret, frame2 = video.read()

        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame1 = np.float32(frame1)

        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2 = np.float32(frame2)

        # Get and filter keypoints
        # This is calculating keypoints in every frame
        # We also want to test without this extensive calculation
        kp = keypoint.kp(cp.copy(frame1))
        kp = filterBorderKeypoints(kp, filterBorder, frame1.shape)

        kp = solver(kp, frame1, frame2, solverNeighbourhood)
        
        # Call interpolation function on the keypoints
        # kp = newkypointsInterpolated(kp, [u,v])

# DEBUG
video = cv2.VideoCapture('../input/input.mp4')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

KLT(video)

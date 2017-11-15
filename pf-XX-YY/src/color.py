import utils
import numpy as np
import cv2

def detectByColor(frame, hsvColor, ballTrace):

    # Transform the color space into HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Gets a black and white image, all the pixels in the color range will be
    # painted white, everything else will be black
    mask = cv2.inRange(hsv, (hsvColor - 20, 25, 0), (hsvColor + 20, 255, 255))

    # Remove some noise from image
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    mask = cv2.erode(mask, kernel, iterations = 2)
    mask = cv2.dilate(mask, kernel, iterations = 1)

    # Gets all the connected components in the mask
    conComponents = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

    # Gets the index of the largest connected component
    componentIndex = filterCentroid(conComponents[3], conComponents[2])

    # Draw in PINK the centroid of all connected components
    #  for i in range(conComponents[3].shape[0]):
    #      cv2.circle(frame, (int(conComponents[3][i][0]), int(conComponents[3][i][1])), 2, (255, 0, 255), -1)

    # Draw in GREEN the centroid of the largest connected component
    cv2.circle(frame, (int(conComponents[3][componentIndex][0]), int(conComponents[3][componentIndex][1])), 4, (0, 255, 0), -1)

    frame, ballTrace = utils.drawBallTrace(frame, (conComponents[3][componentIndex][0], conComponents[3][componentIndex][1]), ballTrace)

    cv2.imshow('Mask', mask)
    cv2.imshow('Frame', frame)

def filterCentroid(centroid, status):

    larger = 0
    largerIndex = -1

    # Goes over all regions selected by connectedComponentsWithStats, selects
    # the one with the larger area.
    # The area for region 'i' is stored in 'status[i][4]'
    for i in range(1, centroid.shape[0]):

        if (status[i][4] > larger):
            larger = status[i][4]
            largerIndex = i

    return largerIndex

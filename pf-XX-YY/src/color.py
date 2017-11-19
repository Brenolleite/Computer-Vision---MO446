import utils
import numpy as np
import cv2

import hough

hsvColor = []

def initColors():
    # Yellow
    hsvColor.append(30)

    # Blue
    hsvColor.append(120)

def filterLargerComponent(areas, selectedLabels):
    # Dont filter if just background present
    if len(areas) <= 1:
        return []

    # Sort in desc order
    largest = sorted(areas, reverse = True)[1]

    # Get all indexes higher than threshold
    indexes = np.where(areas > largest / 3)[0]

    # Get all index different from background
    indexes = indexes[indexes != 0]

    # Get just selected labels from filter
    indexes = np.intersect1d(indexes, selectedLabels)

    return indexes

def houghLabelFilter(circles, labels):
    # Create array of selected colors
    selectedLabels = []

    # Get height and width
    height, width = labels.shape

    # Select all labels that intersects with circles
    for pos in circles:
        x, y, _ = pos

        if x < height and y < width and labels[x][y] != 0:
            selectedLabels.append(labels[x][y])

    return selectedLabels

def detectByColor(frame):
    output = []
    maskJoin = np.zeros((frame.shape[0], frame.shape[1]))

    # Transform the color space into HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for i in range(len(hsvColor)):
        # Gets a black and white image, all the pixels in the color range will be
        # painted white, everything else will be black
        mask = cv2.inRange(hsv, (hsvColor[i] - 40, 86, 6), (hsvColor[i] + 60, 255, 255))

        # Remove some noise from image
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations = 2)
        mask = cv2.dilate(mask, kernel, iterations = 2)

        # Gets all the connected components in the mask
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

        # Fit a circle in the mask
        circles = hough.find(mask)

        # Filter color using hough circles
        selectedLabels = []
        if len(circles) > 0:
            frame = hough.draw(frame, circles)
            selectedLabels = houghLabelFilter(circles, labels)

        # Gets the index of the largest connected component
        # stats[:, 4] gets all the areas of components
        indexes = filterLargerComponent(stats[selectedLabels, 4], selectedLabels)

        # Creating ball information
        for ix in indexes:
            x1, y1, x2, y2, _ = stats[ix]
            output.append((hsvColor[i], x1, y1, x1 + x2, y1 + y2, int(centroids[ix][0]), int(centroids[ix][1])))

        #  maskJoin = cv2.bitwise_or(maskJoin, mask, mask= mask)
        #wName = 'Mask' + str(i)
        #cv2.imshow(wName, mask)
        #cv2.imshow("Hough", frame)

    return output

initColors()

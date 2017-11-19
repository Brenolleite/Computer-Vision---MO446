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

def filterLargerComponent(areas):
    if len(areas) == 1:
        return []
    largest = sorted(areas, reverse = True)[1]
    indexes = np.where(areas > largest / 3)[0]
    indexes = indexes[indexes != 0]

    return indexes

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
        _, pixelsLabel, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

        circles = hough.find(mask)
        if len(circles) > 0:
            frame = hough.draw(frame, circles)
            # label = houghLabelFilter(circles, pixelsLabel)

        # Gets the index of the largest connected component
        # stats[:, 4] gets all the areas of components
        indexes = filterLargerComponent(stats[:,4])

        for ix in indexes:
            x1, y1, x2, y2, _ = stats[ix]
            output.append((hsvColor[i], x1, y1, x1 + x2, y1 + y2, int(centroids[ix][0]), int(centroids[ix][1])))

        #  maskJoin = cv2.bitwise_or(maskJoin, mask, mask= mask)
        wName = 'Mask' + str(i)
        cv2.imshow(wName, mask)
        cv2.imshow("Hough", frame)

    return output

def houghLabelFilter(circles, pixelsLabel):
    selectedLabels = []

    for i in circles[0, :]:
        if i[0] < pixelsLabel.shape[0] and i[1] < pixelsLabel.shape[1]:
            print("Label: ", pixelsLabel[i[0]][i[1]])

initColors()

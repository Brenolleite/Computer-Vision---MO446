import utils
import numpy as np
import cv2

hsvColor = []

def initColors():
    # Red
    hsvColor.append(20)
    hsvColor.append(160)

    # Blue
    hsvColor.append(120)

    # Yellow

def filterLargerComponent(areas, treshold = None):
    if not treshold:
        largest_area = 0
        largest_i = -1

        # Goes over all region's label, skiping the 0 one, since it is background
        for i in range(1, len(areas)):
            if areas[i] > largest_area:
                largest_area = areas[i]
                largest_i = i

        indexes = [largest_i]
    else:
        indexes = np.where(np.logical_and(areas > treshold, areas < treshold * 10))[0]

    return indexes

def detectByColor(frame):
    output = []

    # Transform the color space into HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for i in range(len(hsvColor)):
        # Gets a black and white image, all the pixels in the color range will be
        # painted white, everything else will be black
        mask = cv2.inRange(hsv, (hsvColor[i] - 20, 50, 0), (hsvColor[i] + 20, 255, 255))

        # Remove some noise from image
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations = 1)
        mask = cv2.erode(mask, kernel, iterations = 2)
        mask = cv2.dilate(mask, kernel, iterations = 1)

        # Gets all the connected components in the mask
        _, pixelsLabel, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

        # Gets the index of the largest connected component
        # stats[:, 4] gets all the areas of components
        indexes = filterLargerComponent(stats[:,4], 3000)

        for ix in indexes:
            x1, y1, x2, y2, _ = stats[ix]
            output.append((i, x1, y1, x1 + x2, y1 + y2, int(centroids[ix][0]), int(centroids[ix][1])))

    return output

initColors()
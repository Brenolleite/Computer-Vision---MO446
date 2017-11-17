import utils
import numpy as np
import cv2

hsvColor = 120
output = None

def detectByColor(frame):

    # Transform the color space into HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Gets a black and white image, all the pixels in the color range will be
    # painted white, everything else will be black
    mask = cv2.inRange(hsv, (hsvColor - 20, 50, 0), (hsvColor + 20, 255, 255))

    # Remove some noise from image
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    mask = cv2.erode(mask, kernel, iterations = 2)
    mask = cv2.dilate(mask, kernel, iterations = 1)

    # Gets all the connected components in the mask
    _, pixelsLabel, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

    # Gets the index of the largest connected component
    componentIndex = filterLargerComponent(stats[:,4])

    x1, y1, x2, y2, _ = stats[componentIndex]
    output = (x1, y1, x1 + x2, y1 + y2, int(centroids[componentIndex][0]), int(centroids[componentIndex][1]))

    return output

def filterLargerComponent(status):

    larger = 0
    largerIndex = -1

    # Goes over all region's label, skiping the 0 one, since it is background
    for i in range(1, len(status)):

        if (status[i] > larger):
            larger = status[i]
            largerIndex = i

    return largerIndex

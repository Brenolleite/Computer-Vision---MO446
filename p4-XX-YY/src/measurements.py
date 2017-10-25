import utils
import numpy as np

import cv2

# Wraps each region information in an array of
# [size, mean_color, texture, centroid, bounding_box]
def region_info(img, labels, centroids):

    # Number of regions -1 to ignore the label == -1
    regions = [0] * (len(np.unique(labels)))

    # i = height, k = width
    for i in range(labels.shape[0]):
        for k in range(labels.shape[1]):

            # Index of the region
            index = labels[i][k]

            # Ignoring small regions (index = -1)
            if index == -1:
                continue

            # Creating the info array inside the regions array
            if regions[index] == 0:
                # Create array of [size, mean_color, texture, centroid, bounding_box]
                regions[index] = [0, [0, 0, 0], 0, [0, 0], [9999, 0, 9999, 0]]

            # info array will be changed and then saved into regions array
            info = regions[index]

            # Increase region size [0]
            info[0] += 1

            # Add the total color to calculate the mean color [1]
            info[1] += img[i, k]

            # Get centroid into info [3]
            info[3] = centroids[index]

            # Set bounding box [width1, width2, height1, height2] [4]
            if k < info[4][0]:
                info[4][0] = k
            if k > info[4][1]:
                info[4][1] = k
            if i < info[4][2]:
                info[4][2] = i
            if i > info[4][3]:
                info[4][3] = i

            regions[index] = info

    # Get mean value of colors

    for i in range(1, len(regions)):
        reg = regions[i]
        # Divide total color from number of pixels
        reg[1] = reg[1]/reg[0]


    # Debug
    img = utils.drawBoundingBox(img, regions)

    # Save image
    cv2.imwrite('../output/report_bounding_boxes.jpg', img)

    return regions
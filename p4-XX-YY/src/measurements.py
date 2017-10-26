import numpy as np
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
import cv2

def coccurrence_features(img, bbox):
    # Boundary box [width_min, width_max, height_min, height_max]
    patch = img[bbox[2]:bbox[3], bbox[0]:bbox[1]]

    # Transform to gray scale
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # Get co-occurrence matrix
    glcm = greycomatrix(patch, [1], [0], 256)

    # Get features from co-occurrence matrix
    contrast = greycoprops(glcm, 'contrast')[0][0]
    correlation = greycoprops(glcm, 'correlation')[0][0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0][0]
    energy = greycoprops(glcm, 'energy')[0][0]
    entr = entropy(patch)

    # Set correlation if sqrt not able to suceed
    if np.isnan(correlation):
        correlation = 1

    return [contrast, correlation, dissimilarity, energy, entr]

# Wraps each region information in an array of
# [size, mean_color, texture, centroid, bounding_box]
def region_info(img, components, centroids):

    # Number of regions -1 to ignore the label == -1
    regions = [0] * (len(np.unique(components))-1)

    # i = height, k = width
    for i in range(components.shape[0]):
        for k in range(components.shape[1]):

            # Index of the region
            index = components[i][k]

            # Ignoring small regions (index = -1)
            if index == -1:
                continue

            # Creating the info array inside the regions array
            if regions[index] == 0:
                # Create array of [size, mean_color, texture, centroid, bounding_box]
                regions[index] = [0, [0, 0, 0], [], [0, 0], [9999, 0, 9999, 0]]

            # info array will be changed and then saved into regions array
            info = regions[index]

            # Increase region size [0]
            info[0] += 1

            # Add the total color to calculate the mean color [1]
            info[1] += img[i, k]

            # Store region pixel to correlate afterwards [2]
            info[2].append(int((int(img[i, k][0]) + int(img[i, k][1]) + int(img[i, k][2]))/3))

            # Get centroid into info [3]
            info[3] = centroids[index]

            # Set bounding box
            # [width_min, width_max, height_min, height_max] [4]
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
    # and compute contrast, correlation, and entropy
    for i in range(len(regions)):
        reg = regions[i]
        # Divide total color from number of pixels
        reg[1] = reg[1]/reg[0]

        # Compute co-occurrence features
        # Return [contrast, correlation, dissimilarity, energy, entropy]
        reg[2] = coccurrence_features(img, reg[4])

    return regions
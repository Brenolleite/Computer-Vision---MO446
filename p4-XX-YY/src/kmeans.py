import numpy as np
import cv2
import connected_comp as cc
import utils

def k_means(img):
    # Img prep
    img = img.reshape((-1, 3))
    img = np.float32(img)

    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Number of Clusters
    K = 2

    # Using kmeans on colors intensity
    aux, label, center = cv2.kmeans(img, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return center, label

# Get the mean color of a region, output: (B, G, R)
def getMeanColor(img, labels, index, output):
    # Necessary because we ignored the -1 labels
    index = index + 1

    # Not really a sum, it stores the pixel's color to be averaged in the end
    useless = []
    colorSum = [useless] * len(np.unique(labels))
    print("Unique labels ", np.unique(labels))
    print("Type ", type(colorSum))

    # i = height, k - width
    for i in range(labels.shape[0]):
        for k in range(labels.shape[1]):

            index = labels[i][k]

            if index == -1:
                continue

            colorSum[index].append(img[i][k])


    for i in range(len(colorSum)):
        size = len(colorSum[i])
        aux = np.zeros((1, size, 3), np.uint8)
        print("aux shape ", aux.shape)
        print("colorsum1 ", i, len(colorSum[i]))

        for k in range(size):
            aux[0][k] = (colorSum[i][k][0], colorSum[i][k][1], colorSum[i][k][2])
        #  print("Mean \n", cv2.mean(aux))
        print(np.average(aux, axis = 1))



# Wraps each region information in an array of [size, mean_color, texture, centroid, bounding_box]
def region_info(img, labels, centroids):

    # Number os regions -1 to ignore the label == -1
    regions = [0] * (len(np.unique(labels)) - 1)

    # i = height, k = width
    for i in range(labels.shape[0]):
        for k in range(labels.shape[1]):

            # Index of the region
            index = labels[i][k] - 1

            # Ignoring indexes -1
            if index == -2:
                continue

            # Creating the info array inside the regions array
            if regions[index] == 0:
                regions[index] = [0, 0, 0, 0, [9999, 0, 9999, 0]]

            # info array will be changed and then saved into regions array
            info = regions[index]

            # Increase region size
            info[0] += 1

            # Get the mean color of a region
            # getMeanColor(img, labels, index, info[1])

            # Set bounding box [width1, width2, height1, height2]
            if k < info[4][0]:
                info[4][0] = k
            if k > info[4][1]:
                info[4][1] = k
            if i < info[4][2]:
                info[4][2] = i
            if i > info[4][3]:
                info[4][3] = i

            regions[index] = info

    # Debug
    utils.drawBoundingBox(img, regions)

    return regions

# Debug
img = cv2.imread('../input/boat_1.jpg')

center, label = k_means(img)

kimg = utils.k_image(img, center, label)

cv2.imwrite('../output/teste_kmeans.jpg', kimg)

labels, centroids = cc.conn_comp(kimg, 8)

region_info(img, labels, centroids)

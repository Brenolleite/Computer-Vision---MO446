import cv2
import numpy as np
import copy as cp
import collections as col

import utils

# Clear components which is too small
def clear_components(labels, nlabels):
    counter = np.zeros(nlabels)

    for i in range(nlabels):
        counter[i] = len(labels[labels == i])

    mean = np.mean(counter)

    for i in np.where(counter<mean)[0]:
        labels[labels == i] = -1

    return labels

def dfs_comp(labels, img):
    labels = labels.reshape((img.shape[0], img.shape[1]))

    ctrl = np.zeros(labels.shape)
    components = np.zeros(labels.shape).astype(int)
    compIndex = 1

    for i in range(components.shape[0]):
        for k in range(components.shape[1]):
            if components[i][k] == 0:
                dfs(i, k, labels, ctrl, components, compIndex, labels[i][k])
                compIndex += 1

    print("Regions ", np.unique(int(components)))

def checkPixelForConnection(i, k, currentLabel, labels, ctrl):
    if i > 0 and i < labels.shape[0] and k > 0 and k < labels.shape[1] and labels[i][k] == currentLabel and ctrl[i][k] == 0:
        return True
    return False

def dfs(i, k, labels, ctrl, components, compIndex, currentLabel):

    ctrl[i][k] = 1

    components[i][k] = compIndex

    for m in range(i - 1, i + 1):
        for l in range(k - 1, k + 1):
            if checkPixelForConnection(m, l, currentLabel, labels, ctrl) == True:
                dfs(m, l, labels, ctrl, components, compIndex, labels[i][k])

    return

# Finding connected components
def conn_comp(img, connectivity):
    # Transform image to gray color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Creating all components
    components = np.zeros(img.shape).astype(int)
    centroids_comp = [(-1,-1)]
    comp_count = 1

    B = []
    # Transform to binary image using k-means colors
    for i in np.unique(img):
        img_aux = cp.copy(img)

        # Create binary image
        img_aux[img == i] = 255
        img_aux[img != i] = 0

        cv2.imwrite('../output/back{0}.jpg'.format(i), img_aux)

        # Find connected componnents
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_aux, connectivity, cv2.CV_32S)

        # Remove small componnents found
        labels = clear_components(labels, nlabels)

        out = utils.components_image(labels)
        cv2.imwrite('../output/teste{0}.jpg'.format(i), out)
        #B.append(np.where(labels != 0)[0])

        # Update components counter
        for i in np.unique(labels):
            if i != -1 and i != 0:
                centroids_comp.append(centroids)
                labels[labels == i] = comp_count
                comp_count += 1

        # Updating components
        height, width = labels.shape
        for y in range(height):
            for x in range(width):
                if labels[y,x] != 0:
                    components[y,x] = labels[y,x]

    #print(len(set(B[0]).intersection(B[1])))

    out = utils.components_image(components)
    cv2.imwrite('../output/final.jpg', out)

    return components, centroids_comp

import cv2
import numpy as np
import copy as cp
import collections as col

# Clear components which is too small
def clear_components(components):
    counter = np.zeros(len(np.unique(components)))

    for i in range(len(counter)):
        counter[i] = len(components[components == i])

    mean = np.mean(counter)*10

    for i in np.where(counter<mean)[0]:
        components[components == i] = -1

    return components

# Check neighbors
def checkPixelForConnection(i, j, currentLabel, labels, components):
    return (i >= 0 and i < labels.shape[0] and
            j >= 0 and j < labels.shape[1] and
            labels[i][j] == currentLabel  and
            components[i][j] == -1)

# BFS for connected components
def BFS(labels):
    # Create queue and variables
    queue = []
    compIndex = 0
    components = np.zeros(labels.shape).astype(int) - 1

    # Loop in disconnected components
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if components[i][j] == -1:

                currentLabel = labels[i,j]
                queue.append((i,j))

                # Get all connected components
                while len(queue) > 0:
                    x , y = queue.pop(0)

                    for m in range(x - 1, x + 2):
                        for l in range(y - 1, y + 2):
                            if checkPixelForConnection(m, l, currentLabel, labels, components):
                                components[m, l] = compIndex
                                queue.append((m, l))

                compIndex += 1

    return components

# Finding connected components
def conn_comp(labels, img):
    # Reshaping after kmeans
    labels = labels.reshape((img.shape[0], img.shape[1]))

    # Using BFS for connected components
    components = BFS(labels)

    # Clearing small components
    components = clear_components(components)

    # Update components counter
    comp_count = 0
    for i in np.unique(components):
        if i != -1:
            components[components == i] = comp_count
            comp_count += 1

    return components

# Finding centroids
def centroid(components):
    centroids = np.zeros((len(np.unique(components)), 2))

    for comp in np.unique(components):
        points = []

        for i in range(components.shape[0]):
            for j in range(components.shape[1]):
                if components[i,j] == comp:
                    points.append((j,i))

        points = np.array(points)

        length = points.shape[0]
        sum_x = np.sum(points[:, 0])
        sum_y = np.sum(points[:, 1])
        centroids[comp] = [sum_x/length, sum_y/length]

    return centroids